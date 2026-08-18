"""Periodic FLUX.2 latent decoding for training-time W&B monitoring."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def flow_prediction_to_x0(noisy: Tensor, predicted_flow: Tensor, timestep: Tensor) -> Tensor:
    sigma = timestep.to(device=noisy.device, dtype=noisy.dtype)
    while sigma.ndim < noisy.ndim:
        sigma = sigma.unsqueeze(-1)
    return noisy - predicted_flow.to(dtype=noisy.dtype) * sigma


def unpack_flux2_tokens(tokens: Tensor, vae, *, grid_height: int = 12, grid_width: int = 24) -> Tensor:
    """Invert RoboNana's FLUX patchify + BatchNorm cache representation."""

    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [B, N, C], got {tuple(tokens.shape)}")
    batch, count, packed_channels = tokens.shape
    if count != grid_height * grid_width:
        raise ValueError(f"token count {count} does not match grid {grid_height}x{grid_width}")
    if packed_channels % 4:
        raise ValueError(f"packed channels must be divisible by four, got {packed_channels}")

    packed = tokens.transpose(1, 2).reshape(batch, packed_channels, grid_height, grid_width)
    eps = float(getattr(getattr(vae, "config", None), "batch_norm_eps", 1e-4))
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=packed.device, dtype=packed.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1).to(device=packed.device, dtype=packed.dtype) + eps)
    packed = packed * std + mean

    channels = packed_channels // 4
    raw = packed.reshape(batch, channels, 2, 2, grid_height, grid_width)
    return raw.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels, grid_height * 2, grid_width * 2)


@torch.no_grad()
def decode_flux2_tokens(vae, tokens: Tensor, *, grid_height: int = 12, grid_width: int = 24) -> Tensor:
    raw = unpack_flux2_tokens(tokens, vae, grid_height=grid_height, grid_width=grid_width)
    decoded = vae.decode(raw.to(dtype=next(vae.parameters()).dtype), return_dict=False)[0]
    return decoded.float().add(1.0).div(2.0).clamp(0.0, 1.0)


def should_log_pixel_eval(step: int, interval: int) -> bool:
    return interval > 0 and step > 0 and step % interval == 0


def log_pixel_eval(
    *,
    accelerator,
    step: int,
    current: Tensor,
    target: Tensor,
    prediction: Tensor,
    horizon_idx: int,
) -> None:
    """Upload current/target/prediction and a side-by-side panel to W&B."""

    if not accelerator.is_main_process:
        return
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Pixel eval logging requires wandb") from error

    current_image = current[0].detach().cpu()
    target_image = target[0].detach().cpu()
    prediction_image = prediction[0].detach().cpu()
    panel = torch.cat([current_image, target_image, prediction_image], dim=-1)
    tracker = accelerator.get_tracker("wandb", unwrap=True)
    tracker.log(
        {
            "eval/current_image": wandb.Image(current_image, caption="current frame"),
            "eval/target_future_image": wandb.Image(target_image, caption=f"GT future frame h={horizon_idx}"),
            "eval/predicted_future_image": wandb.Image(prediction_image, caption=f"predicted x0 h={horizon_idx}"),
            "eval/current_gt_prediction": wandb.Image(
                panel,
                caption=f"left=current | middle=GT future | right=predicted future | h={horizon_idx}",
            ),
            "eval/horizon_idx": int(horizon_idx),
        },
        step=int(step),
        commit=False,
    )
