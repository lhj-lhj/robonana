"""Shared flow-matching schedule and Euler update for eval and inference."""

from __future__ import annotations

import torch
from torch import Tensor


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
