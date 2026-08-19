"""Small cache contract shared by preprocessing and the FACT data adapter."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor


CACHE_SCHEMA_VERSION = 1
LANGUAGE_CONTEXT_NAME = "language_context.pt"
LATENT_FOLDER_NAME = "latents"
EPISODE_LANGUAGE_FOLDER_NAME = "language"


def instruction_for_episode(task_dir: str | Path, episode_index: int) -> tuple[str, Path | None]:
    """Return the recorded instruction for one episode, with task fallback."""

    task_dir = Path(task_dir)
    path = task_dir / "instructions" / f"episode{episode_index}.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        for split in ("seen", "unseen"):
            prompts = row.get(split, [])
            if isinstance(prompts, list):
                for prompt in prompts:
                    if isinstance(prompt, str) and prompt.strip():
                        return prompt.strip(), path
    return canonical_instruction(task_dir)


def canonical_instruction(task_dir: str | Path) -> tuple[str, Path | None]:
    """Return one deterministic training instruction from the raw RoboTwin task."""

    task_dir = Path(task_dir)
    instruction_paths = sorted((task_dir / "instructions").glob("episode*.json"))
    for path in instruction_paths:
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        for split in ("seen", "unseen"):
            prompts = row.get(split, [])
            if isinstance(prompts, list):
                for prompt in prompts:
                    if isinstance(prompt, str) and prompt.strip():
                        return prompt.strip(), path
    return task_dir.parent.name.replace("_", " "), None


def episode_cache_path(task_dir: str | Path, episode_index: int) -> Path:
    return Path(task_dir) / "flux_cache" / LATENT_FOLDER_NAME / f"episode_{episode_index:06d}.pt"


def language_context_path(task_dir: str | Path) -> Path:
    return Path(task_dir) / "flux_cache" / LANGUAGE_CONTEXT_NAME


def episode_language_context_path(task_dir: str | Path, episode_index: int) -> Path:
    return (
        Path(task_dir)
        / "flux_cache"
        / EPISODE_LANGUAGE_FOLDER_NAME
        / f"episode_{episode_index:06d}.pt"
    )


def select_current_future_latents(
    frame_latents: Tensor,
    current_index: int,
    horizon_idx: int,
) -> tuple[Tensor, Tensor]:
    """Index one frame cache as ``current_latent`` and ``future_latent_h``.

    Each image is encoded once.  The horizon-conditioned future is selected at
    load time, so caching does not duplicate the same frame for every possible
    ``idx_h``.
    """

    if frame_latents.ndim != 3:
        raise ValueError(f"frame_latents must be [T, image_tokens, channels], got {tuple(frame_latents.shape)}")
    if not 0 <= current_index < frame_latents.shape[0]:
        raise IndexError(f"current_index={current_index} is outside [0, {frame_latents.shape[0]})")
    if horizon_idx < 1:
        raise ValueError(f"horizon_idx must be positive, got {horizon_idx}")
    future_index = min(current_index + horizon_idx, frame_latents.shape[0] - 1)
    return frame_latents[current_index], frame_latents[future_index]
