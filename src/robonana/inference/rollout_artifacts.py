"""Pixel-space helpers for closed-loop RoboTwin world-model rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from world_action_model.image_layouts import ROBOTWIN_VIEW_KEYS


def decoded_frame_to_uint8(frame: torch.Tensor) -> torch.Tensor:
    """Convert one decoded CHW frame from ``[-1, 1]`` to CPU uint8."""

    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError(f"decoded frame must be CHW RGB, got {tuple(frame.shape)}")
    return frame.detach().float().add(1.0).mul(127.5).round().clamp(0, 255).to(
        device="cpu", dtype=torch.uint8
    )


def split_robotwin_composite(
    frame: torch.Tensor,
    *,
    main_view_size: tuple[int, int] = (256, 192),
) -> dict[str, torch.Tensor]:
    """Invert RoboTwin's high-view plus vertically stacked wrist-view canvas."""

    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError(f"composite frame must be CHW RGB, got {tuple(frame.shape)}")
    main_width, main_height = (int(value) for value in main_view_size)
    side_width, side_height = main_width // 2, main_height // 2
    expected_shape = (3, main_height, main_width + side_width)
    if tuple(frame.shape) != expected_shape:
        raise ValueError(f"composite frame must have shape {expected_shape}, got {tuple(frame.shape)}")
    return {
        ROBOTWIN_VIEW_KEYS[0]: frame[:, :, :main_width].contiguous(),
        ROBOTWIN_VIEW_KEYS[1]: frame[:, :side_height, main_width:].contiguous(),
        ROBOTWIN_VIEW_KEYS[2]: frame[:, side_height:, main_width:].contiguous(),
    }


def annotate_rollout_frame(
    frame_uint8: torch.Tensor,
    *,
    trajectory_index: int,
    rollout_index: int,
    rollout_count: int,
    horizon: int,
    value: float,
) -> Image.Image:
    """Overlay the horizon-specific value without modifying the feedback frame."""

    if frame_uint8.dtype != torch.uint8:
        raise TypeError("annotate_rollout_frame expects uint8 input")
    array = frame_uint8.permute(1, 2, 0).contiguous().numpy()
    image = Image.fromarray(np.asarray(array), mode="RGB")
    draw = ImageDraw.Draw(image)
    label = (
        f"trajectory={trajectory_index:02d} rollout={rollout_index + 1}/{rollout_count} "
        f"h={horizon:02d} value={float(value):.5f}"
    )
    draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return image
