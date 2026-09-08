"""Shared flow-matching schedule and Euler update for eval and inference."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, TYPE_CHECKING

import torch
from torch import Tensor

from robonana.models.position_ids import image_position_ids, text_position_ids

if TYPE_CHECKING:
    from robonana.models.flux2_scalar_expert import FrozenFluxKVCache


@dataclass(frozen=True)
class MacWorldSample:
    future: Tensor
    future_state: Tensor
    reward_logits: Tensor
    success_logit: Tensor


@dataclass(frozen=True)
class QRejectionSample:
    action: Tensor
    candidates: Tensor
    candidate_q: Tensor
    best_index: Tensor
    condition_cache: FrozenFluxKVCache | None = None


@dataclass(frozen=True)
class MacImaginaryRollout:
    selected_action: Tensor
    candidates: Tensor
    candidate_q: Tensor
    best_index: Tensor
    future: Tensor
    future_state: Tensor
    reward_logits: Tensor
    success_logit: Tensor
    chunk_return: Tensor
    target_next_value: Tensor
    online_next_value: Tensor
    value_target_return: Tensor
    q_target_return: Tensor
    condition_cache: FrozenFluxKVCache | None = None


def sample_action_flow(
    *,
    action_noise: Tensor,
    schedule: Tensor,
    predict_action: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Denoise an action chunk from pure noise with the shared Flow-Euler path."""

    if schedule.ndim != 1 or schedule.numel() < 2:
        raise ValueError("schedule must contain at least a start and end sigma")
    if not bool(torch.isclose(schedule[0], schedule.new_tensor(1.0))):
        raise ValueError("schedule must start at sigma=1 pure noise")
    if not bool(torch.isclose(schedule[-1], schedule.new_tensor(0.0))):
        raise ValueError("schedule must end at sigma=0 clean data")
    if bool(torch.any(schedule[1:] > schedule[:-1])):
        raise ValueError("schedule must be monotonically decreasing")
    sampled_action = action_noise
    for sigma, sigma_next in zip(schedule[:-1], schedule[1:]):
        action_velocity = predict_action(sampled_action, sigma)
        sampled_action = flow_euler_step(sampled_action, action_velocity, sigma, sigma_next)
    return sampled_action


def flow_euler_schedule(
    num_inference_steps: int,
    *,
    flow_shift: float,
    device: torch.device | str,
) -> Tensor:
    """Return the inference sigma path from pure noise (1) to clean data (0)."""

    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if flow_shift <= 0:
        raise ValueError("flow_shift must be positive")
    sigma = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
    if flow_shift != 1.0:
        sigma = flow_shift * sigma / (1.0 + (flow_shift - 1.0) * sigma)
    return sigma


def flow_euler_step(sample: Tensor, velocity: Tensor, sigma: Tensor, sigma_next: Tensor) -> Tensor:
    """Integrate ``dx/dsigma = velocity`` for one decreasing-sigma Euler step."""

    delta = (sigma_next - sigma).to(device=sample.device, dtype=sample.dtype)
    return sample + delta * velocity.to(device=sample.device, dtype=sample.dtype)


def _as_batch_horizon(
    chunk_horizon: int | Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    horizon = torch.as_tensor(chunk_horizon, device=device, dtype=torch.long)
    if horizon.ndim == 0:
        horizon = horizon.expand(batch_size)
    if horizon.ndim not in (1, 2) or horizon.shape[0] != batch_size:
        raise ValueError("chunk_horizon must be scalar, [B], or [B,K]")
    return horizon


def sample_flux2_action(
    *,
    model,
    context: Tensor,
    current_latents: Tensor,
    state: Tensor,
    context_mask: Tensor,
    action_noise: Tensor,
    chunk_horizon: int | Tensor,
    schedule: Tensor,
    grid_height: int,
    grid_width: int,
) -> Tensor:
    """Sample Stage-1 actions through the shared training/inference path.

    Maintained MAC calls use this for live policy inference and online
    candidate generation; non-MAC models are rejected.
    """

    if action_noise.ndim != 3:
        raise ValueError("action_noise must have shape [B,T,action_dim]")
    batch_size = action_noise.shape[0]
    if context.shape[0] != batch_size or current_latents.shape[0] != batch_size:
        raise ValueError("action sampling inputs must share one batch dimension")
    device = action_noise.device
    horizon = _as_batch_horizon(chunk_horizon, batch_size=batch_size, device=device)
    if horizon.ndim != 1:
        raise ValueError("Stage-1 action sampling requires one horizon per sample")
    context_ids = text_position_ids(batch_size, context.shape[1], device)
    current_ids = image_position_ids(
        batch_size,
        grid_height=grid_height,
        grid_width=grid_width,
        time_coord=torch.zeros_like(horizon),
        device=device,
    )
    model_spec = getattr(model, "module", model)
    if getattr(model_spec, "architecture_version", None) != "mac_mot_v2":
        raise ValueError("action sampling requires a mac_mot_v2 model")
    if not bool(torch.all(horizon == 48)) or action_noise.shape[1] != 48:
        raise ValueError("mac_mot_v2 action sampling requires a full 48-step chunk")
    cache = model_spec.prefill_condition_cache(
        context=context, context_ids=context_ids, current_latents=current_latents,
        current_ids=current_ids, state=state, context_mask=context_mask,
    )
    indices = torch.arange(batch_size, device=device)
    return sample_action_flow(
        action_noise=action_noise, schedule=schedule,
        predict_action=lambda action, sigma: model_spec.predict_action_cached(
            cache, action, batch_indices=indices, timestep=sigma
        ),
    )


def evaluate_mac_critics(
    *,
    model,
    context: Tensor,
    current_latents: Tensor,
    state: Tensor,
    context_mask: Tensor,
    clean_action: Tensor,
    grid_height: int,
    grid_width: int,
    condition_cache: FrozenFluxKVCache | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate deterministic ``Value(s)`` and ``Q(s, action_chunk)``."""

    model_spec = getattr(model, "module", model)
    if getattr(model_spec, "architecture_version", None) != "mac_mot_v2":
        raise ValueError("deterministic critics require a mac_mot_v2 model")
    batch_size = context.shape[0]
    device = context.device
    context_ids = text_position_ids(batch_size, context.shape[1], device)
    current_ids = image_position_ids(
        batch_size,
        grid_height=grid_height,
        grid_width=grid_width,
        time_coord=torch.zeros(batch_size, device=device, dtype=torch.long),
        device=device,
    )
    empty_ids = torch.empty(batch_size, 0, 4, device=device, dtype=torch.long)
    empty_image = current_latents.new_empty(batch_size, 0, current_latents.shape[-1])
    empty_state = state.new_empty(batch_size, 0, state.shape[-1])
    empty_action = clean_action.new_empty(batch_size, 0, clean_action.shape[-1])
    empty_scalar = clean_action.new_empty(batch_size, 0, 1)
    zeros = torch.zeros(batch_size, device=device, dtype=torch.float32)
    horizon = torch.full(
        (batch_size,), int(model_spec.chunk_horizon), device=device, dtype=torch.long
    )
    common = dict(
        context=context,
        context_ids=context_ids,
        current_latents=current_latents,
        current_ids=current_ids,
        noisy_future_latents=empty_image,
        future_ids=empty_ids,
        state=state,
        noisy_pred_action=empty_action,
        gt_action_cond=clean_action,
        chunk_horizon=horizon,
        noisy_future_state=empty_state,
        noisy_reward=empty_scalar,
        noisy_q=empty_scalar,
        action_timestep=zeros,
        wm_timestep=zeros,
        context_mask=context_mask,
    )
    return model(**common, critic_kind="both", condition_cache=condition_cache)


def evaluate_mac_target_value(
    *,
    model,
    target_value_expert,
    context: Tensor,
    current_latents: Tensor,
    state: Tensor,
    context_mask: Tensor,
    grid_height: int,
    grid_width: int,
    cache=None,
) -> Tensor:
    """Evaluate the detached EMA Value expert on the single frozen FLUX."""

    model_spec = getattr(model, "module", model)
    batch = context.shape[0]
    device = context.device
    context_ids = text_position_ids(batch, context.shape[1], device)
    current_ids = image_position_ids(
        batch,
        grid_height=grid_height,
        grid_width=grid_width,
        time_coord=torch.zeros(batch, device=device, dtype=torch.long),
        device=device,
    )
    return model_spec.predict_value(
        context=context,
        context_ids=context_ids,
        current_latents=current_latents,
        current_ids=current_ids,
        state=state,
        context_mask=context_mask,
        expert=target_value_expert,
        cache=cache,
    )


@torch.no_grad()
def sample_mac_world(
    *,
    model,
    context: Tensor,
    current_latents: Tensor,
    state: Tensor,
    context_mask: Tensor,
    clean_action: Tensor,
    future_noise: Tensor,
    future_state_noise: Tensor,
    schedule: Tensor,
    grid_height: int,
    grid_width: int,
    condition_cache: FrozenFluxKVCache | None = None,
    use_cache: bool = True,
) -> MacWorldSample:
    """Generate one fixed-chunk transition with invariant-prefix reuse.

    ``use_cache=False`` is the independent full-forward numerical oracle.
    This sampler is inference-only; stage-1 training still uses full autograd.
    """

    model_spec = getattr(model, "module", model)
    if getattr(model_spec, "architecture_version", None) != "mac_mot_v2":
        raise ValueError("imaginary world rollout requires a mac_mot_v2 model")
    if schedule.ndim != 1 or schedule.numel() < 2:
        raise ValueError("schedule must contain at least a start and end sigma")
    if not bool(torch.isclose(schedule[0], schedule.new_tensor(1.0))):
        raise ValueError("schedule must start at sigma=1")
    if not bool(torch.isclose(schedule[-1], schedule.new_tensor(0.0))):
        raise ValueError("schedule must end at sigma=0")
    batch_size = context.shape[0]
    device = context.device
    context_ids = text_position_ids(batch_size, context.shape[1], device)
    current_ids = image_position_ids(
        batch_size,
        grid_height=grid_height,
        grid_width=grid_width,
        time_coord=torch.zeros(batch_size, device=device, dtype=torch.long),
        device=device,
    )
    horizon = torch.full(
        (batch_size,), int(model_spec.chunk_horizon), device=device, dtype=torch.long
    )
    future_ids = image_position_ids(
        batch_size,
        grid_height=grid_height,
        grid_width=grid_width,
        time_coord=horizon,
        device=device,
    )
    empty_action = clean_action.new_empty(batch_size, 0, clean_action.shape[-1])
    empty_scalar = clean_action.new_empty(batch_size, 0, 1)
    zeros = torch.zeros(batch_size, device=device, dtype=torch.float32)
    sampled_future = future_noise
    sampled_future_state = future_state_noise
    world_cache = None
    if use_cache:
        if not model_spec.condition_cache_compatible(condition_cache):
            condition_cache = prefill_mac_condition(
                model=model_spec, context=context, current_latents=current_latents,
                state=state, context_mask=context_mask,
                grid_height=grid_height, grid_width=grid_width)
        world_cache = model_spec.prefill_world_cache(
            condition_cache=condition_cache, clean_action=clean_action,
            language_length=context.shape[1], state_length=state.shape[1],
            image_length=current_latents.shape[1], future_state_length=future_state_noise.shape[1],
            future_image_length=future_noise.shape[1], context_mask=context_mask)

    def predict(sampled_image: Tensor, sampled_state: Tensor, sigma: Tensor):
        if world_cache is not None:
            return model_spec.predict_world_cached(
                world_cache, noisy_future_latents=sampled_image, noisy_future_state=sampled_state,
                future_ids=future_ids, wm_timestep=sigma.expand(batch_size))
        return model(
            context=context,
            context_ids=context_ids,
            current_latents=current_latents,
            current_ids=current_ids,
            noisy_future_latents=sampled_image,
            future_ids=future_ids,
            state=state,
            noisy_pred_action=empty_action,
            gt_action_cond=clean_action,
            chunk_horizon=horizon,
            noisy_future_state=sampled_state,
            noisy_reward=empty_scalar,
            noisy_q=empty_scalar,
            action_timestep=zeros,
            wm_timestep=sigma.expand(batch_size),
            context_mask=context_mask,
        )

    for sigma, sigma_next in zip(schedule[:-1], schedule[1:]):
        output = predict(sampled_future, sampled_future_state, sigma)
        sampled_future = flow_euler_step(
            sampled_future, output.image, sigma, sigma_next
        )
        sampled_future_state = flow_euler_step(
            sampled_future_state, output.future_state, sigma, sigma_next
        )
    # R/U never read S'/I' or world sigma. Their prefill logits are already
    # final; no 21st full forward is needed after the 20 velocity evaluations.
    final = world_cache if world_cache is not None else predict(sampled_future, sampled_future_state, schedule[-1])
    return MacWorldSample(
        future=sampled_future,
        future_state=sampled_future_state,
        reward_logits=final.reward,
        success_logit=final.success,
    )


def prefill_mac_condition(*, model, context, current_latents, state, context_mask,
                          grid_height, grid_width):
    """Build one request-local clean C cache, shared across candidates/steps."""
    batch = context.shape[0]
    return model.prefill_condition_cache(
        context=context,
        context_ids=text_position_ids(batch, context.shape[1], context.device),
        current_latents=current_latents,
        current_ids=image_position_ids(
            batch, grid_height=grid_height, grid_width=grid_width,
            time_coord=torch.zeros(batch, device=context.device, dtype=torch.long),
            device=context.device,
        ),
        state=state, context_mask=context_mask,
    )


@torch.no_grad()
def sample_q_rejection(
    *,
    model,
    context: Tensor,
    current_latents: Tensor,
    state: Tensor,
    context_mask: Tensor,
    candidate_count: int,
    action_noise: Tensor,
    schedule: Tensor,
    grid_height: int,
    grid_width: int,
    candidate_batch_size: int | None = None,
    return_condition_cache: bool = False,
) -> QRejectionSample:
    """Sample independent BC chunks and select the highest-Q candidate.

    This is the deterministic rejection rule used by MAC.  Candidate actions
    are generated with the shared FLUX prefix and Q is evaluated only after
    the full candidate set has been denoised.
    """

    candidate_count = int(candidate_count)
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    batch_size = context.shape[0]
    model_spec = getattr(model, "module", model)
    expected = (
        batch_size,
        candidate_count,
        int(model_spec.chunk_horizon),
        int(model_spec.action_dim),
    )
    if tuple(action_noise.shape) != expected:
        raise ValueError(f"action_noise must have shape {expected}")

    group_size = int(
        candidate_batch_size
        if candidate_batch_size is not None
        else os.environ.get("ROBONANA_REJECTION_CANDIDATE_BATCH_SIZE", "8")
    )
    if group_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    cache = prefill_mac_condition(
        model=model_spec, context=context, current_latents=current_latents,
        state=state, context_mask=context_mask,
        grid_height=grid_height, grid_width=grid_width,
    )
    groups = []
    for start in range(0, candidate_count, group_size):
        noise = action_noise[:, start:start + group_size]
        width = noise.shape[1]
        indices = torch.arange(batch_size, device=noise.device).repeat_interleave(width)
        actions = sample_action_flow(
            action_noise=noise.reshape(batch_size * width, model_spec.chunk_horizon, model_spec.action_dim),
            schedule=schedule,
            predict_action=lambda action, sigma: model_spec.predict_action_cached(
                cache, action, batch_indices=indices, timestep=sigma
            ),
        )
        groups.append(actions.reshape(batch_size, width, model_spec.chunk_horizon, model_spec.action_dim))
    candidates = torch.cat(groups, dim=1)
    # Q-only scoring. C is shared with policy sampling, and G is re-encoded
    # with its clean segment/time convention instead of reusing noisy A K/V.
    candidate_q = model_spec.score_q_candidates(
        cache, candidates, candidate_batch_size=group_size,
    )
    best_index = candidate_q.argmax(dim=1)
    batch_indices = torch.arange(batch_size, device=candidate_q.device)
    return QRejectionSample(
        action=candidates[batch_indices, best_index],
        candidates=candidates,
        candidate_q=candidate_q,
        best_index=best_index,
        condition_cache=cache if return_condition_cache else None,
    )


@torch.no_grad()
def generate_mac_imaginary_rollout_h1(
    *,
    online_model,
    target_value_expert,
    context: Tensor,
    current_latents: Tensor,
    state: Tensor,
    context_mask: Tensor,
    candidate_count: int,
    action_noise: Tensor,
    future_noise: Tensor,
    future_state_noise: Tensor,
    schedule: Tensor,
    discount: float,
    reward_non_goal: float,
    reward_goal: float,
    return_scale: float,
    grid_height: int,
    grid_width: int,
) -> MacImaginaryRollout:
    """Generate a fresh one-chunk on-policy imaginary transition."""

    if not 0.0 < float(discount) <= 1.0:
        raise ValueError("discount must lie in (0, 1]")
    if float(return_scale) <= 0:
        raise ValueError("return_scale must be positive")
    rejection = sample_q_rejection(
        model=online_model,
        context=context,
        current_latents=current_latents,
        state=state,
        context_mask=context_mask,
        candidate_count=candidate_count,
        return_condition_cache=True,
        action_noise=action_noise,
        schedule=schedule,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    world = sample_mac_world(
        model=online_model,
        context=context,
        current_latents=current_latents,
        state=state,
        context_mask=context_mask,
        clean_action=rejection.action,
        condition_cache=rejection.condition_cache,
        future_noise=future_noise,
        future_state_noise=future_state_noise,
        schedule=schedule,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    model_spec = getattr(online_model, "module", online_model)
    next_cache = prefill_mac_condition(
        model=model_spec, context=context, current_latents=world.future,
        state=world.future_state, context_mask=context_mask,
        grid_height=grid_height, grid_width=grid_width,
    )
    target_next_value_normalized = evaluate_mac_target_value(
        model=online_model,
        target_value_expert=target_value_expert,
        context=context,
        current_latents=world.future,
        state=world.future_state,
        context_mask=context_mask,
        grid_height=grid_height,
        grid_width=grid_width,
        cache=next_cache,
    )
    batch = context.shape[0]
    context_ids = text_position_ids(batch, context.shape[1], context.device)
    current_ids = image_position_ids(
        batch,
        grid_height=grid_height,
        grid_width=grid_width,
        time_coord=torch.zeros(batch, device=context.device, dtype=torch.long),
        device=context.device,
    )
    online_next_value_normalized = model_spec.predict_value(
        context=context,
        context_ids=context_ids,
        current_latents=world.future,
        current_ids=current_ids,
        state=world.future_state,
        context_mask=context_mask,
        cache=next_cache,
    )
    probabilities = world.reward_logits.float().sigmoid()
    predicted_rewards = float(reward_non_goal) + probabilities * (
        float(reward_goal) - float(reward_non_goal)
    )
    offsets = torch.arange(
        int(model_spec.chunk_horizon),
        device=predicted_rewards.device,
        dtype=torch.float32,
    )
    discounts = torch.pow(
        torch.full_like(offsets, float(discount)), offsets
    )
    chunk_return = (predicted_rewards * discounts[None]).sum(dim=1, keepdim=True)
    # MAC uses a discrete terminate mask.  The world model's sigmoid is
    # thresholded and detached; critics cannot exploit soft terminal leakage.
    bootstrap_mask = (world.success_logit.float().sigmoid() < 0.5).float()
    target_next_value = target_next_value_normalized.float() * float(return_scale)
    online_next_value = online_next_value_normalized.float() * float(return_scale)
    gamma_chunk = float(discount) ** int(model_spec.chunk_horizon)
    value_target = chunk_return + gamma_chunk * bootstrap_mask * target_next_value
    # Original MAC has no target Q.  Its Q target uses stop-gradient online V:
    # https://github.com/kwanyoungpark/MAC/blob/main/agents/mac.py#L191-L217
    q_target = chunk_return + gamma_chunk * bootstrap_mask * online_next_value
    return MacImaginaryRollout(
        selected_action=rejection.action.detach(),
        candidates=rejection.candidates.detach(),
        candidate_q=rejection.candidate_q.detach(),
        best_index=rejection.best_index.detach(),
        future=world.future.detach(),
        future_state=world.future_state.detach(),
        reward_logits=world.reward_logits.detach(),
        success_logit=world.success_logit.detach(),
        chunk_return=chunk_return.detach(),
        target_next_value=target_next_value.detach(),
        online_next_value=online_next_value.detach(),
        value_target_return=value_target.detach(),
        q_target_return=q_target.detach(),
        condition_cache=rejection.condition_cache,
    )
