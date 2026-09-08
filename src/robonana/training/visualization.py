"""Shared FLUX.2 latent decoding for current world-model reports."""

from __future__ import annotations

import torch
from torch import Tensor


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
