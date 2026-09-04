"""Pixel-space helpers for closed-loop RoboTwin world-model rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from world_action_model.image_layouts import ROBOTWIN_VIEW_KEYS


RECORDED_VIDEO_SIZE = (320, 240)
RECORDED_OVERLAY_HEIGHT = 42
_RECORDED_OVERLAY_FONT = ImageFont.load_default()


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
    reward: float,
    q: float,
) -> Image.Image:
    """Overlay horizon reward/Q without modifying the feedback frame."""

    if frame_uint8.dtype != torch.uint8:
        raise TypeError("annotate_rollout_frame expects uint8 input")
    array = frame_uint8.permute(1, 2, 0).contiguous().numpy()
    image = Image.fromarray(np.asarray(array), mode="RGB")
    draw = ImageDraw.Draw(image)
    label = (
        f"trajectory={trajectory_index:02d} rollout={rollout_index + 1}/{rollout_count} "
        f"h={horizon:02d} reward={float(reward):.5f} Q={float(q):.5f}"
    )
    draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return image


def recorded_frame_chunk_horizon(
    frame_index: int,
    *,
    action_chunk: int,
) -> tuple[int | None, int]:
    """Map an observation frame to the action chunk that predicted it.

    Frame zero is the initial observation and has no preceding action.  For
    later frames, chunk ``c`` starts from observation ``c * action_chunk`` and
    predicts the next ``action_chunk`` observations at horizons ``1..T``.
    """

    frame_index = int(frame_index)
    action_chunk = int(action_chunk)
    if frame_index < 0:
        raise ValueError("frame_index cannot be negative")
    if action_chunk <= 0:
        raise ValueError("action_chunk must be positive")
    if frame_index == 0:
        return None, 0
    preceding_transition = frame_index - 1
    return preceding_transition // action_chunk, preceding_transition % action_chunk + 1


def annotate_recorded_frame(
    frame_uint8: torch.Tensor,
    *,
    group: str,
    episode_index: int,
    frame_index: int,
    action_chunk: int,
    reward: float | None,
    q: float | None,
) -> Image.Image:
    """Overlay source identity and the per-frame Stage-2 return prediction."""

    if frame_uint8.dtype != torch.uint8:
        raise TypeError("annotate_recorded_frame expects uint8 input")
    if frame_uint8.ndim != 3 or frame_uint8.shape[0] != 3:
        raise ValueError("annotate_recorded_frame expects CHW RGB input")
    chunk_index, horizon = recorded_frame_chunk_horizon(
        frame_index,
        action_chunk=action_chunk,
    )
    image = Image.fromarray(
        frame_uint8.permute(1, 2, 0).contiguous().numpy(),
        mode="RGB",
    )
    if image.size != RECORDED_VIDEO_SIZE:
        image = image.resize(RECORDED_VIDEO_SIZE, resample=Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    identity = f"group={group} episode={episode_index:03d} frame={frame_index:04d}"
    if chunk_index is None:
        returns = "chunk=--- h=00 current-frame (no preceding action)"
    else:
        returns = (
            f"chunk={chunk_index + 1:03d} h={horizon:02d}/{action_chunk:02d} "
            f"reward_h={float(reward):.5f} Q_h={float(q):.5f}"
        )
    draw.rectangle(
        (0, 0, image.width - 1, RECORDED_OVERLAY_HEIGHT - 1),
        fill=(0, 0, 0),
    )
    draw.text((6, 4), identity, fill=(255, 255, 255), font=_RECORDED_OVERLAY_FONT)
    draw.text((6, 22), returns, fill=(255, 255, 255), font=_RECORDED_OVERLAY_FONT)
    return image
