"""Small loss helpers compatible with FACT failure masks."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_mse(prediction: Tensor, target: Tensor, sample_mask: Tensor | None = None) -> Tensor:
    per_sample = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    if sample_mask is None:
        return per_sample.mean()
    mask = sample_mask.to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
    return (per_sample * mask).sum() / mask.sum().clamp_min(1e-8)


def masked_action_mse(
    prediction: Tensor, target: Tensor, step_mask: Tensor, success_mask: Tensor
) -> Tensor:
    """Success-only BC on real actions, excluding absorbing padding steps."""
    per_step = (prediction.float() - target.float()).square().mean(dim=-1)
    if per_step.shape != step_mask.shape:
        raise ValueError("action_valid_mask must match [batch, action_horizon]")
    valid = step_mask.to(per_step)
    per_sample = (per_step * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    success = success_mask.to(per_sample).reshape(-1)
    return (per_sample * success).sum() / success.sum().clamp_min(1)


def masked_bce_with_logits(
    logits: Tensor,
    target: Tensor,
    sample_mask: Tensor | None = None,
) -> Tensor:
    per_sample = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none"
    ).flatten(1).mean(dim=1)
    if sample_mask is None:
        return per_sample.mean()
    mask = sample_mask.to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
    return (per_sample * mask).sum() / mask.sum().clamp_min(1e-8)


def masked_elementwise_bce_with_logits(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """BCE normalized by valid chunk positions rather than padded positions."""

    if logits.shape != target.shape or logits.shape != valid_mask.shape:
        raise ValueError(
            "logits, target, and valid_mask must have identical shapes, got "
            f"{tuple(logits.shape)}, {tuple(target.shape)}, {tuple(valid_mask.shape)}"
        )
    elementwise = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none"
    )
    mask = valid_mask.to(device=elementwise.device, dtype=elementwise.dtype)
    return (elementwise * mask).sum() / mask.sum().clamp_min(1.0)


def deterministic_return_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    return_scale: float,
    sample_mask: Tensor | None = None,
) -> Tensor:
    """MSE for deterministic Value/Q heads using one fixed return scale."""

    return_scale = float(return_scale)
    if return_scale <= 0:
        raise ValueError("return_scale must be positive")
    normalized_target = target.float() / return_scale
    return masked_mse(prediction, normalized_target, sample_mask)
