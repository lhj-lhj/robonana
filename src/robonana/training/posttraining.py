"""Iterative RoboNana posttraining helpers built on the shared FLUX model."""

from __future__ import annotations

import copy
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor, nn

from robonana.sampling import (
    flow_euler_schedule,
    sample_flux2_action,
    sample_flux2_world,
)


@dataclass(frozen=True)
class CandidateSearchResult:
    pseudo_action: Tensor
    candidates: Tensor
    candidate_q: Tensor
    best_index: Tensor
    best_q: Tensor
    behavior_q: Tensor
    elapsed_ms: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class TDTargetResult:
    q_target: Tensor
    next_q: Tensor
    next_action: Tensor
    bootstrap_mask: Tensor
    q_loss_mask: Tensor
    discount_factor: Tensor
    elapsed_ms: float
    peak_memory_bytes: int


def _cuda_peak_start(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def _cuda_peak_finish(device: torch.device, baseline: int) -> int:
    if device.type != "cuda":
        return 0
    torch.cuda.synchronize(device)
    return max(0, int(torch.cuda.max_memory_allocated(device)) - int(baseline))


@contextmanager
def evaluating(model: nn.Module) -> Iterator[None]:
    """Temporarily switch a model to eval without changing its gradients."""

    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type in {"cuda", "cpu"} and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


class FullModelEMA:
    """A callable, rank-local FP32 EMA copy of the trainable Flux2FACTModel."""

    def __init__(
        self,
        online_model: nn.Module,
        *,
        decay: float = 0.995,
        update_every_optimizer_steps: int = 1,
        start_step: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must lie in [0,1)")
        if int(update_every_optimizer_steps) <= 0:
            raise ValueError("EMA update interval must be positive")
        if int(start_step) < 0:
            raise ValueError("EMA start_step cannot be negative")
        self.decay = float(decay)
        self.update_every_optimizer_steps = int(update_every_optimizer_steps)
        self.start_step = int(start_step)
        self.update_count = 0
        self.last_online_l2 = 0.0
        self.model = copy.deepcopy(online_model)
        if device is not None:
            self.model.to(device)
        self.model.float().eval().requires_grad_(False)

    @torch.no_grad()
    def exact_copy_from(self, online_model: nn.Module) -> None:
        online_state = online_model.state_dict()
        ema_state = self.model.state_dict()
        if online_state.keys() != ema_state.keys():
            raise RuntimeError("online and EMA model state keys differ")
        for name, target in ema_state.items():
            source = online_state[name]
            if isinstance(target, Tensor):
                target.copy_(source.detach().to(device=target.device, dtype=target.dtype))
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def update(
        self,
        online_model: nn.Module,
        *,
        optimizer_step: int,
        optimizer_step_succeeded: bool,
    ) -> bool:
        if not optimizer_step_succeeded:
            return False
        optimizer_step = int(optimizer_step)
        if optimizer_step < self.start_step:
            return False
        if (optimizer_step - self.start_step) % self.update_every_optimizer_steps:
            return False
        online_parameters = dict(online_model.named_parameters())
        squared = torch.zeros((), device=next(self.model.parameters()).device, dtype=torch.float64)
        count = 0
        for name, ema_parameter in self.model.named_parameters():
            online_parameter = online_parameters[name].detach().to(
                device=ema_parameter.device, dtype=torch.float32
            )
            ema_parameter.mul_(self.decay).add_(online_parameter, alpha=1.0 - self.decay)
            difference = ema_parameter.double() - online_parameter.double()
            squared.add_(difference.square().sum())
            count += difference.numel()
        online_buffers = dict(online_model.named_buffers())
        for name, ema_buffer in self.model.named_buffers():
            ema_buffer.copy_(online_buffers[name].detach().to(ema_buffer))
        self.update_count += 1
        self.last_online_l2 = float(torch.sqrt(squared / max(count, 1)).cpu().item())
        self.model.eval().requires_grad_(False)
        return True

    def state_dict(self) -> dict[str, Tensor]:
        return {
            name: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for name, value in self.model.state_dict().items()
            if isinstance(value, Tensor)
        }

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        destination = self.model.state_dict()
        if destination.keys() != state_dict.keys():
            missing = sorted(destination.keys() - state_dict.keys())
            unexpected = sorted(state_dict.keys() - destination.keys())
            raise RuntimeError(f"EMA checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        converted = {
            name: value.to(device=destination[name].device, dtype=destination[name].dtype)
            for name, value in state_dict.items()
        }
        self.model.load_state_dict(converted, strict=True)
        self.model.eval().requires_grad_(False)

    def assert_not_in_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        overlap = [name for name, parameter in self.model.named_parameters() if id(parameter) in optimizer_ids]
        if overlap:
            raise RuntimeError(f"EMA parameters leaked into optimizer groups: {overlap[:5]}")


def _repeat_candidates(value: Tensor, candidate_count: int) -> Tensor:
    return value.repeat_interleave(int(candidate_count), dim=0)


def _sample_actions_microbatched(
    *,
    model: nn.Module,
    context: Tensor,
    current_latents: Tensor,
    state: Tensor,
    context_mask: Tensor,
    action_noise: Tensor,
    horizon: int,
    schedule: Tensor,
    grid_height: int,
    grid_width: int,
    microbatch_size: int,
    autocast_dtype: torch.dtype | None = None,
) -> Tensor:
    outputs = []
    for start in range(0, action_noise.shape[0], int(microbatch_size)):
        stop = min(start + int(microbatch_size), action_noise.shape[0])
        context_manager = (
            _autocast(action_noise.device, autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with context_manager:
            outputs.append(
                sample_flux2_action(
                    model=model,
                    context=context[start:stop],
                    current_latents=current_latents[start:stop],
                    state=state[start:stop],
                    context_mask=context_mask[start:stop],
                    action_noise=action_noise[start:stop],
                    horizon_idx=horizon,
                    schedule=schedule,
                    grid_height=grid_height,
                    grid_width=grid_width,
                )
            )
    return torch.cat(outputs, dim=0)


def _sample_q_microbatched(
    *,
    model: nn.Module,
    context: Tensor,
    current_latents: Tensor,
    state: Tensor,
    context_mask: Tensor,
    clean_action: Tensor,
    horizon: int,
    future_state_noise: Tensor,
    reward_template: Tensor,
    q_noise: Tensor,
    schedule: Tensor,
    grid_height: int,
    grid_width: int,
    microbatch_size: int,
    autocast_dtype: torch.dtype,
) -> Tensor:
    outputs = []
    latent_channels = current_latents.shape[-1]
    for start in range(0, clean_action.shape[0], int(microbatch_size)):
        stop = min(start + int(microbatch_size), clean_action.shape[0])
        future_noise = current_latents.new_empty(stop - start, 0, latent_channels)
        with _autocast(clean_action.device, autocast_dtype):
            world = sample_flux2_world(
                model=model,
                context=context[start:stop],
                current_latents=current_latents[start:stop],
                state=state[start:stop],
                context_mask=context_mask[start:stop],
                clean_action=clean_action[start:stop],
                horizon_idx=horizon,
                future_noise=future_noise,
                future_state_noise=future_state_noise[start:stop],
                reward_template=reward_template[start:stop],
                q_noise=q_noise[start:stop],
                schedule=schedule,
                grid_height=grid_height,
                grid_width=grid_width,
            )
        outputs.append(world.q.float())
    return torch.cat(outputs, dim=0).reshape(clean_action.shape[0], -1)[:, 0]


@torch.no_grad()
def search_failure_candidates(
    *,
    online_model: nn.Module,
    ema_model: nn.Module,
    context: Tensor,
    current_latents: Tensor,
    state: Tensor,
    context_mask: Tensor,
    behavior_action: Tensor,
    candidate_count: int = 8,
    candidate_horizon: int = 48,
    action_sampling_steps: int = 20,
    q_sampling_steps: int = 20,
    flow_shift: float = 1.0,
    microbatch_size: int = 16,
    grid_height: int = 12,
    grid_width: int = 24,
    ema_autocast_dtype: torch.dtype = torch.bfloat16,
    action_noise: Tensor | None = None,
    common_future_state_noise: Tensor | None = None,
    common_reward_template: Tensor | None = None,
    common_q_noise: Tensor | None = None,
) -> CandidateSearchResult:
    """Online best-of-N actions ranked by EMA Q with common world noise."""

    if int(candidate_count) != 8:
        raise ValueError("the first posttraining version requires candidate_count=8")
    if int(candidate_horizon) != 48:
        raise ValueError("candidate ranking must use idx_h=48")
    if behavior_action.ndim != 3 or behavior_action.shape[1] != candidate_horizon:
        raise ValueError("behavior_action must contain one complete 48-step chunk")
    if int(microbatch_size) <= 0:
        raise ValueError("candidate_microbatch_size must be positive")
    device = behavior_action.device
    batch_size, action_horizon, action_dim = behavior_action.shape
    started = time.perf_counter()
    baseline = _cuda_peak_start(device)
    if action_noise is None:
        action_noise = torch.randn(
            batch_size,
            candidate_count,
            action_horizon,
            action_dim,
            device=device,
            dtype=behavior_action.dtype,
        )
    expected_noise_shape = (batch_size, candidate_count, action_horizon, action_dim)
    if tuple(action_noise.shape) != expected_noise_shape:
        raise ValueError(f"action_noise must have shape {expected_noise_shape}")

    expanded_context = _repeat_candidates(context, candidate_count)
    expanded_current = _repeat_candidates(current_latents, candidate_count)
    expanded_state = _repeat_candidates(state, candidate_count)
    expanded_context_mask = _repeat_candidates(context_mask, candidate_count)
    action_schedule = flow_euler_schedule(
        action_sampling_steps, flow_shift=flow_shift, device=device
    )
    with evaluating(online_model):
        candidates = _sample_actions_microbatched(
            model=online_model,
            context=expanded_context,
            current_latents=expanded_current,
            state=expanded_state,
            context_mask=expanded_context_mask,
            action_noise=action_noise.reshape(batch_size * candidate_count, action_horizon, action_dim),
            horizon=candidate_horizon,
            schedule=action_schedule,
            grid_height=grid_height,
            grid_width=grid_width,
            microbatch_size=microbatch_size,
        ).reshape(batch_size, candidate_count, action_horizon, action_dim)
    candidates = candidates.detach()

    common_future_state_noise = (
        torch.randn(batch_size, 1, state.shape[-1], device=device, dtype=behavior_action.dtype)
        if common_future_state_noise is None
        else common_future_state_noise
    )
    common_reward_template = (
        torch.zeros(batch_size, 1, 1, device=device, dtype=behavior_action.dtype)
        if common_reward_template is None
        else common_reward_template
    )
    common_q_noise = (
        torch.randn(batch_size, 1, 1, device=device, dtype=behavior_action.dtype)
        if common_q_noise is None
        else common_q_noise
    )
    world_schedule = flow_euler_schedule(q_sampling_steps, flow_shift=flow_shift, device=device)
    ema_model.eval()
    flat_candidate_q = _sample_q_microbatched(
        model=ema_model,
        context=expanded_context,
        current_latents=expanded_current,
        state=expanded_state,
        context_mask=expanded_context_mask,
        clean_action=candidates.reshape(batch_size * candidate_count, action_horizon, action_dim),
        horizon=candidate_horizon,
        future_state_noise=_repeat_candidates(common_future_state_noise, candidate_count),
        reward_template=_repeat_candidates(common_reward_template, candidate_count),
        q_noise=_repeat_candidates(common_q_noise, candidate_count),
        schedule=world_schedule,
        grid_height=grid_height,
        grid_width=grid_width,
        microbatch_size=microbatch_size,
        autocast_dtype=ema_autocast_dtype,
    )
    candidate_q = flat_candidate_q.reshape(batch_size, candidate_count)
    behavior_q = _sample_q_microbatched(
        model=ema_model,
        context=context,
        current_latents=current_latents,
        state=state,
        context_mask=context_mask,
        clean_action=behavior_action,
        horizon=candidate_horizon,
        future_state_noise=common_future_state_noise,
        reward_template=common_reward_template,
        q_noise=common_q_noise,
        schedule=world_schedule,
        grid_height=grid_height,
        grid_width=grid_width,
        microbatch_size=microbatch_size,
        autocast_dtype=ema_autocast_dtype,
    )
    best_index = candidate_q.argmax(dim=1)
    batch_index = torch.arange(batch_size, device=device)
    pseudo_action = candidates[batch_index, best_index].detach()
    best_q = candidate_q[batch_index, best_index]
    return CandidateSearchResult(
        pseudo_action=pseudo_action,
        candidates=candidates,
        candidate_q=candidate_q.detach(),
        best_index=best_index.detach(),
        best_q=best_q.detach(),
        behavior_q=behavior_q.detach(),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        peak_memory_bytes=_cuda_peak_finish(device, baseline),
    )


@torch.no_grad()
def build_td_targets(
    *,
    ema_model: nn.Module,
    context: Tensor,
    next_current_latents: Tensor,
    next_state: Tensor,
    context_mask: Tensor,
    reward_h: Tensor,
    delta_steps: Tensor,
    success_terminal_h: Tensor,
    action_template: Tensor,
    discount: float = 0.999,
    target_action_horizon: int = 48,
    action_sampling_steps: int = 20,
    q_sampling_steps: int = 20,
    flow_shift: float = 1.0,
    grid_height: int = 12,
    grid_width: int = 24,
    microbatch_size: int = 16,
    ema_autocast_dtype: torch.dtype = torch.bfloat16,
    next_action_noise: Tensor | None = None,
) -> TDTargetResult:
    """Build stop-gradient TD targets; only success terminals stop bootstrap."""

    if int(target_action_horizon) != 48:
        raise ValueError("TD next-action horizon must be 48")
    device = reward_h.device
    batch_size = reward_h.shape[0]
    reward_h = reward_h.reshape(batch_size, 1, 1).float()
    delta_steps = delta_steps.reshape(batch_size).long()
    success_terminal_h = success_terminal_h.reshape(batch_size).float()
    q_loss_mask = (delta_steps > 0).float()
    bootstrap_mask = 1.0 - success_terminal_h
    valid_bootstrap = (q_loss_mask > 0) & (bootstrap_mask > 0)
    discount_factor = torch.pow(
        torch.full((batch_size,), float(discount), device=device, dtype=torch.float32),
        delta_steps.float(),
    )
    next_q = torch.zeros(batch_size, device=device, dtype=torch.float32)
    next_action = torch.zeros_like(action_template)
    started = time.perf_counter()
    baseline = _cuda_peak_start(device)

    if bool(valid_bootstrap.any()):
        selected = valid_bootstrap.nonzero(as_tuple=False).reshape(-1)
        selected_action_noise = (
            torch.randn_like(action_template[selected])
            if next_action_noise is None
            else next_action_noise[selected]
        )
        action_schedule = flow_euler_schedule(
            action_sampling_steps, flow_shift=flow_shift, device=device
        )
        ema_model.eval()
        sampled_next_action = _sample_actions_microbatched(
            model=ema_model,
            context=context[selected],
            current_latents=next_current_latents[selected],
            state=next_state[selected],
            context_mask=context_mask[selected],
            action_noise=selected_action_noise,
            horizon=target_action_horizon,
            schedule=action_schedule,
            grid_height=grid_height,
            grid_width=grid_width,
            microbatch_size=microbatch_size,
            autocast_dtype=ema_autocast_dtype,
        ).detach()
        next_action[selected] = sampled_next_action
        q_schedule = flow_euler_schedule(q_sampling_steps, flow_shift=flow_shift, device=device)
        selected_next_q = _sample_q_microbatched(
            model=ema_model,
            context=context[selected],
            current_latents=next_current_latents[selected],
            state=next_state[selected],
            context_mask=context_mask[selected],
            clean_action=sampled_next_action,
            horizon=target_action_horizon,
            future_state_noise=torch.randn(
                selected.numel(), 1, next_state.shape[-1], device=device, dtype=action_template.dtype
            ),
            reward_template=torch.zeros(
                selected.numel(), 1, 1, device=device, dtype=action_template.dtype
            ),
            q_noise=torch.randn(
                selected.numel(), 1, 1, device=device, dtype=action_template.dtype
            ),
            schedule=q_schedule,
            grid_height=grid_height,
            grid_width=grid_width,
            microbatch_size=microbatch_size,
            autocast_dtype=ema_autocast_dtype,
        )
        next_q[selected] = selected_next_q

    q_target = reward_h.reshape(batch_size) + discount_factor * bootstrap_mask * next_q
    return TDTargetResult(
        q_target=q_target.reshape(batch_size, 1, 1).detach(),
        next_q=next_q.reshape(batch_size, 1, 1).detach(),
        next_action=next_action.detach(),
        bootstrap_mask=bootstrap_mask.detach(),
        q_loss_mask=q_loss_mask.detach(),
        discount_factor=discount_factor.detach(),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        peak_memory_bytes=_cuda_peak_finish(device, baseline),
    )
