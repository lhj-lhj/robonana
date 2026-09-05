"""Small loss helpers compatible with FACT failure masks."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from robonana.models.flux2_fact import Flux2FACTOutput


def masked_mse(prediction: Tensor, target: Tensor, sample_mask: Tensor | None = None) -> Tensor:
    per_sample = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    if sample_mask is None:
        return per_sample.mean()
    mask = sample_mask.to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
    return (per_sample * mask).sum() / mask.sum().clamp_min(1e-8)


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


def joint_flow_loss(
    output: Flux2FACTOutput,
    *,
    image_target: Tensor,
    action_target: Tensor,
    future_state_target: Tensor,
    reward_target: Tensor,
    success_target: Tensor,
    q_target: Tensor,
    dino_target: Tensor | None = None,
    action_loss_mask: Tensor | None = None,
    q_loss_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    losses = {
        "image_loss": masked_mse(output.image, image_target),
        "action_loss": masked_mse(output.action, action_target, action_loss_mask),
        "future_state_loss": masked_mse(output.future_state, future_state_target),
        "reward_loss": masked_bce_with_logits(output.reward, reward_target),
        "success_loss": masked_bce_with_logits(output.success, success_target),
        "q_loss": masked_mse(output.q, q_target, q_loss_mask),
    }
    if dino_target is not None:
        if output.dino is None:
            raise ValueError("dino_target was provided but the model produced no DINO output")
        losses["dino_loss"] = masked_mse(output.dino, dino_target)
    return losses
