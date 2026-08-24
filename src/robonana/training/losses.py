"""Small loss helpers compatible with FACT failure masks."""

from __future__ import annotations

import torch
from torch import Tensor

from robonana.models.flux2_fact import Flux2FACTOutput


def masked_mse(prediction: Tensor, target: Tensor, sample_mask: Tensor | None = None) -> Tensor:
    per_sample = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    if sample_mask is None:
        return per_sample.mean()
    mask = sample_mask.to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
    return (per_sample * mask).sum() / mask.sum().clamp_min(1e-8)


def joint_flow_loss(
    output: Flux2FACTOutput,
    *,
    image_target: Tensor,
    action_target: Tensor,
    future_state_target: Tensor,
    value_target: Tensor,
    dino_target: Tensor | None = None,
    action_loss_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    losses = {
        "image_loss": masked_mse(output.image, image_target),
        "action_loss": masked_mse(output.action, action_target, action_loss_mask),
        "future_state_loss": masked_mse(output.future_state, future_state_target),
        "value_loss": masked_mse(output.value, value_target),
    }
    if dino_target is not None:
        if output.dino is None:
            raise ValueError("dino_target was provided but the model produced no DINO output")
        losses["dino_loss"] = masked_mse(output.dino, dino_target)
    return losses
