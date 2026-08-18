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
    targets: Tensor,
    predictions: Tensor,
    horizons: Tensor,
) -> None:
    """Upload one two-row panel for fixed horizons of the same current frame."""

    if not accelerator.is_main_process:
        return
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Pixel eval logging requires wandb") from error

    current_image = current[0].detach().cpu()
    target_images = targets.detach().cpu()
    prediction_images = predictions.detach().cpu()
    horizon_values = [int(value) for value in horizons.detach().cpu().reshape(-1).tolist()]
    if target_images.shape[0] != len(horizon_values) or prediction_images.shape[0] != len(horizon_values):
        raise ValueError("targets, predictions, and horizons must have the same leading length")
    top_row = torch.cat([current_image, *target_images.unbind(0)], dim=-1)
    bottom_row = torch.cat([current_image, *prediction_images.unbind(0)], dim=-1)
    panel = torch.cat([top_row, bottom_row], dim=-2)
    horizon_caption = " | ".join(f"h={value}" for value in horizon_values)
    tracker = accelerator.get_tracker("wandb", unwrap=True)
    tracker.log(
        {
            "eval/fixed_horizon_grid": wandb.Image(
                panel,
                caption=(
                    f"columns: current | {horizon_caption}; "
                    "top row: current + GT futures; bottom row: current + predicted futures"
                ),
            ),
            "eval/fixed_horizons": ",".join(str(value) for value in horizon_values),
        },
        step=int(step),
        commit=False,
    )
