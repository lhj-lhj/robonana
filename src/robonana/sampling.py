"""Shared flow-matching schedule and Euler update for eval and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class TwoStageFlowSample:
    action: Tensor
    future: Tensor
    future_state: Tensor
    reward: Tensor
    q: Tensor


@dataclass(frozen=True)
class WorldFlowSample:
    future: Tensor
    future_state: Tensor
    reward: Tensor
    q: Tensor


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


def sample_world_flow(
    *,
    clean_action: Tensor,
    future_noise: Tensor,
    future_state_noise: Tensor,
    reward_noise: Tensor,
    q_noise: Tensor,
    schedule: Tensor,
    predict_world: Callable[
        [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor],
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
    sampled_reward = reward_noise
    sampled_q = q_noise
    for sigma, sigma_next in zip(schedule[:-1], schedule[1:]):
        image_velocity, state_velocity, reward_velocity, q_velocity = predict_world(
            sampled_future,
            sampled_future_state,
            sampled_reward,
            sampled_q,
            clean_action,
            sigma,
        )
        sampled_future = flow_euler_step(sampled_future, image_velocity, sigma, sigma_next)
        sampled_future_state = flow_euler_step(
            sampled_future_state, state_velocity, sigma, sigma_next
        )
        sampled_reward = flow_euler_step(
            sampled_reward, reward_velocity, sigma, sigma_next
        )
        sampled_q = flow_euler_step(sampled_q, q_velocity, sigma, sigma_next)

    return WorldFlowSample(
        future=sampled_future,
        future_state=sampled_future_state,
        reward=sampled_reward,
        q=sampled_q,
    )


def sample_two_stage_flow(
    *,
    action_noise: Tensor,
    future_noise: Tensor,
    future_state_noise: Tensor,
    reward_noise: Tensor,
    q_noise: Tensor,
    schedule: Tensor,
    predict_action: Callable[[Tensor, Tensor], Tensor],
    predict_world: Callable[
        [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor],
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
        reward_noise=reward_noise,
        q_noise=q_noise,
        schedule=schedule,
        predict_world=predict_world,
    )

    return TwoStageFlowSample(
        action=sampled_action,
        future=world.future,
        future_state=world.future_state,
        reward=world.reward,
        q=world.q,
    )
