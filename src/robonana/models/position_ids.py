"""FLUX.2 position IDs shared by training and online inference."""

from __future__ import annotations

import torch
from torch import Tensor


def text_position_ids(batch_size: int, length: int, device: torch.device) -> Tensor:
    ids = torch.zeros(batch_size, length, 4, device=device, dtype=torch.long)
    ids[:, :, 3] = torch.arange(length, device=device)
    return ids


def image_position_ids(
    batch_size: int,
    *,
    grid_height: int,
    grid_width: int,
    time_coord: Tensor,
    device: torch.device,
) -> Tensor:
    height = torch.arange(grid_height, device=device)
    width = torch.arange(grid_width, device=device)
    spatial = torch.cartesian_prod(height, width)
    ids = torch.zeros(
        batch_size,
        grid_height * grid_width,
        4,
        device=device,
        dtype=torch.long,
    )
    ids[:, :, 0] = time_coord.to(device=device, dtype=torch.long).reshape(batch_size, 1)
    ids[:, :, 1:3] = spatial[None]
    return ids
