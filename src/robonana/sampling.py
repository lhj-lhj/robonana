"""Shared flow-matching schedule and Euler update for eval and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from robonana.models.position_ids import image_position_ids, text_position_ids


@dataclass(frozen=True)
class TwoStageFlowSample:
    action: Tensor
    future: Tensor
    future_state: Tensor
    reward: Tensor
    success: Tensor
    q: Tensor


@dataclass(frozen=True)
class WorldFlowSample:
    future: Tensor
    future_state: Tensor
    reward: Tensor
    success: Tensor
    q: Tensor


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
    next_value: Tensor
    target_return: Tensor


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
    horizon_idx: int | Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    horizon = torch.as_tensor(horizon_idx, device=device, dtype=torch.long)
    if horizon.ndim == 0:
        horizon = horizon.expand(batch_size)
    if horizon.ndim not in (1, 2) or horizon.shape[0] != batch_size:
        raise ValueError("horizon_idx must be scalar, [B], or [B,K]")
    return horizon


def sample_flux2_action(
    *,
    model,
    context: Tensor,
    current_latents: Tensor,
    state: Tensor,
    context_mask: Tensor,
    action_noise: Tensor,
    horizon_idx: int | Tensor,
    schedule: Tensor,
    grid_height: int,
    grid_width: int,
) -> Tensor:
    """Sample Stage-1 actions for either online or EMA RoboNana.

    This is the shared model-facing path used by live inference, failure
    candidate generation, and TD next-action generation.
    """

    if action_noise.ndim != 3:
        raise ValueError("action_noise must have shape [B,T,action_dim]")
    batch_size = action_noise.shape[0]
    if context.shape[0] != batch_size or current_latents.shape[0] != batch_size:
        raise ValueError("action sampling inputs must share one batch dimension")
    device = action_noise.device
    horizon = _as_batch_horizon(horizon_idx, batch_size=batch_size, device=device)
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
    empty_ids = torch.empty(batch_size, 0, 4, device=device, dtype=torch.long)
    empty_image = current_latents.new_empty(batch_size, 0, current_latents.shape[-1])
    empty_state = state.new_empty(batch_size, 0, state.shape[-1])
    empty_scalar = action_noise.new_empty(batch_size, 0, 1)
    clean_gt_action = torch.zeros_like(action_noise)
    clean_wm_time = torch.zeros(batch_size, device=device, dtype=torch.float32)

    def predict_action(sampled_action: Tensor, sigma: Tensor) -> Tensor:
        output = model(
            context=context,
            context_ids=context_ids,
            current_latents=current_latents,
            current_ids=current_ids,
            noisy_future_latents=empty_image,
            future_ids=empty_ids,
            state=state,
            noisy_pred_action=sampled_action,
            gt_action_cond=clean_gt_action,
            horizon_idx=horizon,
            noisy_future_state=empty_state,
            noisy_reward=empty_scalar,
            noisy_q=empty_scalar,
            action_timestep=sigma.expand(batch_size),
            wm_timestep=clean_wm_time,
            context_mask=context_mask,
        )
        return output.action

    return sample_action_flow(
        action_noise=action_noise,
        schedule=schedule,
        predict_action=predict_action,
    )


def sample_flux2_world(
    *,
    model,
    context: Tensor,
    current_latents: Tensor,
    state: Tensor,
    context_mask: Tensor,
    clean_action: Tensor,
    horizon_idx: int | Tensor,
    future_noise: Tensor,
    future_state_noise: Tensor,
    reward_template: Tensor,
    q_noise: Tensor,
    schedule: Tensor,
    grid_height: int,
    grid_width: int,
) -> WorldFlowSample:
    """Sample Stage-2 world/Q flow plus direct reward/success outputs."""

    batch_size = clean_action.shape[0]
    device = clean_action.device
    horizon = _as_batch_horizon(horizon_idx, batch_size=batch_size, device=device)
    packed = horizon.ndim == 2
    if packed:
        if future_noise.ndim != 4 or future_noise.shape[:2] != horizon.shape:
            raise ValueError("packed future_noise must have shape [B,K,N,C]")
        token_count = future_noise.shape[2]
        if token_count:
            flat_horizon = horizon.reshape(-1)
            future_ids = image_position_ids(
                batch_size * horizon.shape[1],
                grid_height=grid_height,
                grid_width=grid_width,
                time_coord=flat_horizon,
                device=device,
            ).reshape(batch_size, horizon.shape[1], token_count, 4)
        else:
            future_ids = torch.empty(
                batch_size, horizon.shape[1], 0, 4, device=device, dtype=torch.long
            )
    else:
        if future_noise.ndim != 3 or future_noise.shape[0] != batch_size:
            raise ValueError("future_noise must have shape [B,N,C]")
        token_count = future_noise.shape[1]
        if token_count:
            future_ids = image_position_ids(
                batch_size,
                grid_height=grid_height,
                grid_width=grid_width,
                time_coord=horizon,
                device=device,
            )
        else:
            future_ids = torch.empty(batch_size, 0, 4, device=device, dtype=torch.long)
    context_ids = text_position_ids(batch_size, context.shape[1], device)
    current_ids = image_position_ids(
        batch_size,
        grid_height=grid_height,
        grid_width=grid_width,
        time_coord=torch.zeros(batch_size, device=device, dtype=torch.long),
        device=device,
    )
    clean_action_time = torch.zeros(batch_size, device=device, dtype=torch.float32)
    empty_pred_action = clean_action.new_empty(batch_size, 0, clean_action.shape[-1])

    def predict_world(
        sampled_future: Tensor,
        sampled_future_state: Tensor,
        reward_query: Tensor,
        sampled_q: Tensor,
        sampled_action: Tensor,
        sigma: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        output = model(
            context=context,
            context_ids=context_ids,
            current_latents=current_latents,
            current_ids=current_ids,
            noisy_future_latents=sampled_future,
            future_ids=future_ids,
            state=state,
            noisy_pred_action=empty_pred_action,
            gt_action_cond=sampled_action,
            horizon_idx=horizon,
            noisy_future_state=sampled_future_state,
            noisy_reward=reward_query,
            noisy_q=sampled_q,
            action_timestep=clean_action_time,
            wm_timestep=sigma.expand(batch_size),
            context_mask=context_mask,
        )
        return output.image, output.future_state, output.reward, output.success, output.q

    return sample_world_flow(
        clean_action=clean_action,
        future_noise=future_noise,
        future_state_noise=future_state_noise,
        reward_template=reward_template,
        q_noise=q_noise,
        schedule=schedule,
        predict_world=predict_world,
    )


def sample_world_flow(
    *,
    clean_action: Tensor,
    future_noise: Tensor,
    future_state_noise: Tensor,
    reward_template: Tensor,
    q_noise: Tensor,
    schedule: Tensor,
    predict_world: Callable[
        [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ],
) -> WorldFlowSample:
    """Denoise world targets from pure noise under a fixed clean action chunk."""

    if schedule.ndim != 1 or schedule.numel() < 2:
        raise ValueError("schedule must contain at least a start and end sigma")
    if not bool(torch.isclose(schedule[0], schedule.new_tensor(1.0))):
        raise ValueError("schedule must start at sigma=1 pure noise")
    if not bool(torch.isclose(schedule[-1], schedule.new_tensor(0.0))):
        raise ValueError("schedule must end at sigma=0 clean data")
    if bool(torch.any(schedule[1:] > schedule[:-1])):
        raise ValueError("schedule must be monotonically decreasing")

    sampled_future = future_noise
    sampled_future_state = future_state_noise
    sampled_q = q_noise
    for sigma, sigma_next in zip(schedule[:-1], schedule[1:]):
        image_velocity, state_velocity, _, _, q_velocity = predict_world(
            sampled_future,
            sampled_future_state,
            reward_template,
            sampled_q,
            clean_action,
            sigma,
        )
        sampled_future = flow_euler_step(sampled_future, image_velocity, sigma, sigma_next)
        sampled_future_state = flow_euler_step(
            sampled_future_state, state_velocity, sigma, sigma_next
        )
        sampled_q = flow_euler_step(sampled_q, q_velocity, sigma, sigma_next)

    # Reward and success are direct heads.  Evaluate them once on the fully
    # denoised world/Q state instead of integrating them through the flow.
    _, _, direct_reward, success_logit, _ = predict_world(
        sampled_future,
        sampled_future_state,
        reward_template,
        sampled_q,
        clean_action,
        schedule[-1],
    )

    return WorldFlowSample(
        future=sampled_future,
        future_state=sampled_future_state,
        reward=direct_reward,
        success=success_logit,
        q=sampled_q,
    )


def sample_two_stage_flow(
    *,
    action_noise: Tensor,
    future_noise: Tensor,
    future_state_noise: Tensor,
    reward_template: Tensor,
    q_noise: Tensor,
    schedule: Tensor,
    predict_action: Callable[[Tensor, Tensor], Tensor],
    predict_world: Callable[
        [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ],
) -> TwoStageFlowSample:
    """Run FACT-style action-first, world-second inference from pure noise."""

    if schedule.ndim != 1 or schedule.numel() < 2:
        raise ValueError("schedule must contain at least a start and end sigma")
    if not bool(torch.isclose(schedule[0], schedule.new_tensor(1.0))):
        raise ValueError("schedule must start at sigma=1 pure noise")
    if not bool(torch.isclose(schedule[-1], schedule.new_tensor(0.0))):
        raise ValueError("schedule must end at sigma=0 clean data")
    if bool(torch.any(schedule[1:] > schedule[:-1])):
        raise ValueError("schedule must be monotonically decreasing")
    sampled_action = sample_action_flow(
        action_noise=action_noise,
        schedule=schedule,
        predict_action=predict_action,
    )

    world = sample_world_flow(
        clean_action=sampled_action,
        future_noise=future_noise,
        future_state_noise=future_state_noise,
        reward_template=reward_template,
        q_noise=q_noise,
        schedule=schedule,
        predict_world=predict_world,
    )

    return TwoStageFlowSample(
        action=sampled_action,
        future=world.future,
        future_state=world.future_state,
        reward=world.reward,
        success=world.success,
        q=world.q,
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
) -> tuple[Tensor, Tensor]:
    """Evaluate deterministic ``Value(s)`` and ``Q(s, action_chunk)``."""

    model_spec = getattr(model, "module", model)
    if getattr(model_spec, "architecture_version", None) != "mac_v1":
        raise ValueError("deterministic critics require a mac_v1 model")
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
    output = model(
        context=context,
        context_ids=context_ids,
        current_latents=current_latents,
        current_ids=current_ids,
        noisy_future_latents=empty_image,
        future_ids=empty_ids,
        state=state,
        noisy_pred_action=empty_action,
        gt_action_cond=clean_action,
        horizon_idx=horizon,
        noisy_future_state=empty_state,
        noisy_reward=empty_scalar,
        noisy_q=empty_scalar,
        action_timestep=zeros,
        wm_timestep=zeros,
        context_mask=context_mask,
    )
    if output.value is None:
        raise RuntimeError("mac_v1 model did not produce Value")
    return output.value, output.q


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
) -> MacWorldSample:
    """Generate one learned fixed-chunk world transition."""

    model_spec = getattr(model, "module", model)
    if getattr(model_spec, "architecture_version", None) != "mac_v1":
        raise ValueError("imaginary world rollout requires a mac_v1 model")
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

    def predict(sampled_image: Tensor, sampled_state: Tensor, sigma: Tensor):
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
            horizon_idx=horizon,
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
    final = predict(sampled_future, sampled_future_state, schedule[-1])
    return MacWorldSample(
        future=sampled_future,
        future_state=sampled_future_state,
        reward_logits=final.reward,
        success_logit=final.success,
    )


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
) -> QRejectionSample:
    """Sample independent BC chunks and return deterministic-Q argmax."""

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

    def repeat(value: Tensor) -> Tensor:
        return value[:, None].expand(-1, candidate_count, *value.shape[1:]).reshape(
            batch_size * candidate_count, *value.shape[1:]
        )

    flat_noise = action_noise.reshape(
        batch_size * candidate_count, model_spec.chunk_horizon, model_spec.action_dim
    )
    candidates = sample_flux2_action(
        model=model,
        context=repeat(context),
        current_latents=repeat(current_latents),
        state=repeat(state),
        context_mask=repeat(context_mask),
        action_noise=flat_noise,
        horizon_idx=int(model_spec.chunk_horizon),
        schedule=schedule,
        grid_height=grid_height,
        grid_width=grid_width,
    ).reshape(
        batch_size, candidate_count, model_spec.chunk_horizon, model_spec.action_dim
    )
    _, flat_q = evaluate_mac_critics(
        model=model,
        context=repeat(context),
        current_latents=repeat(current_latents),
        state=repeat(state),
        context_mask=repeat(context_mask),
        clean_action=candidates.reshape(
            batch_size * candidate_count, model_spec.chunk_horizon, model_spec.action_dim
        ),
        grid_height=grid_height,
        grid_width=grid_width,
    )
    candidate_q = flat_q.reshape(batch_size, candidate_count)
    best_index = candidate_q.argmax(dim=1)
    batch_indices = torch.arange(batch_size, device=candidate_q.device)
    return QRejectionSample(
        action=candidates[batch_indices, best_index],
        candidates=candidates,
        candidate_q=candidate_q,
        best_index=best_index,
    )


@torch.no_grad()
def generate_mac_imaginary_rollout_h1(
    *,
    online_model,
    ema_model,
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
        future_noise=future_noise,
        future_state_noise=future_state_noise,
        schedule=schedule,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    empty_action = rejection.action.new_empty(
        rejection.action.shape[0], 0, rejection.action.shape[-1]
    )
    next_value_normalized, _ = evaluate_mac_critics(
        model=ema_model,
        context=context,
        current_latents=world.future,
        state=world.future_state,
        context_mask=context_mask,
        clean_action=empty_action,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    probabilities = world.reward_logits.float().sigmoid()
    predicted_rewards = float(reward_non_goal) + probabilities * (
        float(reward_goal) - float(reward_non_goal)
    )
    online_spec = getattr(online_model, "module", online_model)
    offsets = torch.arange(
        int(online_spec.chunk_horizon),
        device=predicted_rewards.device,
        dtype=torch.float32,
    )
    discounts = torch.pow(
        torch.full_like(offsets, float(discount)), offsets
    )
    chunk_return = (predicted_rewards * discounts[None]).sum(dim=1, keepdim=True)
    terminal_probability = world.success_logit.float().sigmoid()
    next_value = next_value_normalized.float() * float(return_scale)
    target_return = chunk_return + (
        float(discount) ** int(online_spec.chunk_horizon)
    ) * (1.0 - terminal_probability) * next_value
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
        next_value=next_value.detach(),
        target_return=target_return.detach(),
    )
