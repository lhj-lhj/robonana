"""Streaming RoboTwin normalization statistics and episode index generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from .robotwin_hdf5 import ALOHA_DELTA_MASK, EpisodeRecord, discover_episode_records
from .robotwin_lerobot import DEFAULT_TASK_GLOBS, discover_lerobot_episode_records


def _episode_index_row(record: EpisodeRecord, root: Path) -> dict:
    """Serialize all replay-selection metadata without merging physical pools."""

    return {
        "task_name": record.task_name,
        "task_dir": str(record.task_dir.relative_to(root)),
        "source": str(record.source.relative_to(root)),
        "episode_index": record.episode_index,
        "length": record.length,
        "success": record.success,
        "failure_episode": not record.success,
        "round_id": record.round_id,
        "policy_checkpoint": record.policy_checkpoint,
        "policy_version": record.policy_version,
        "has_final_observation": record.has_final_observation,
        "time_limit_truncated": record.time_limit_truncated,
    }


class RunningMoments:
    def __init__(self, dim: int) -> None:
        self.count = 0
        self.total = np.zeros(dim, dtype=np.float64)
        self.total_square = np.zeros(dim, dtype=np.float64)
        self.minimum = np.full(dim, np.inf, dtype=np.float64)
        self.maximum = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.total.size)
        self.count += values.shape[0]
        self.total += values.sum(axis=0)
        self.total_square += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def as_dict(self) -> dict[str, list[float]]:
        if self.count == 0:
            raise RuntimeError("cannot finalize empty statistics")
        mean = self.total / self.count
        variance = np.maximum(self.total_square / self.count - np.square(mean), 0.0)
        return {
            "mean": mean.tolist(),
            "std": np.sqrt(variance).clip(min=1e-8).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


def compute_robotwin_metadata(
    records: Iterable[EpisodeRecord],
    *,
    dataset_root: str | Path,
    action_chunk: int = 48,
    action_dim: int = 14,
) -> tuple[dict, dict]:
    root = Path(dataset_root).expanduser().resolve()
    records = list(records)
    if action_dim <= 0 or action_dim > ALOHA_DELTA_MASK.size:
        raise ValueError(f"action_dim must lie in [1, {ALOHA_DELTA_MASK.size}]")
    state_moments = RunningMoments(action_dim)
    action_moments = RunningMoments(action_dim)
    delta_mask = ALOHA_DELTA_MASK[:action_dim]

    for record in records:
        with h5py.File(record.source, "r") as handle:
            vector = np.asarray(handle["joint_action/vector"], dtype=np.float32)[:, :action_dim]
            action_key = "policy_action/vector" if "policy_action/vector" in handle else "joint_action/vector"
            policy_action = np.asarray(handle[action_key], dtype=np.float32)[:, :action_dim]
        state_moments.update(vector)
        time = np.arange(record.length, dtype=np.int64)[:, None]
        offsets = np.arange(action_chunk, dtype=np.int64)[None]
        chunk = policy_action[np.clip(time + offsets, 0, record.length - 1)].copy()
        chunk[:, :, delta_mask] -= vector[:, None, delta_mask]
        action_moments.update(chunk)

    index = {
        "schema_version": 1,
        "episodes": [_episode_index_row(record, root) for record in records],
    }
    stats = {
        "schema_version": 1,
        "action_representation": "ALOHA joint delta except grippers",
        "action_chunk": int(action_chunk),
        "norm_stats": {
            "observation.state": state_moments.as_dict(),
            "action": action_moments.as_dict(),
            "value": {"min": [-1.0], "max": [2.0]},
        },
    }
    return index, stats


def write_robotwin_metadata(
    dataset_root: str | Path,
    *,
    task_glob: str = "*/aloha-agilex_clean_50",
    action_chunk: int = 48,
    action_dim: int = 14,
    index_path: str | Path | None = None,
    stats_path: str | Path | None = None,
) -> tuple[Path, Path]:
    root = Path(dataset_root).expanduser().resolve()
    records = discover_episode_records(root, task_glob)
    index, stats = compute_robotwin_metadata(
        records,
        dataset_root=root,
        action_chunk=action_chunk,
        action_dim=action_dim,
    )
    index_output = Path(index_path).expanduser() if index_path else root / "robonana_index.json"
    stats_output = Path(stats_path).expanduser() if stats_path else root / "robonana_norm_stats.json"
    for path, payload in ((index_output, index), (stats_output, stats)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    return index_output, stats_output


def compute_robotwin_lerobot_metadata(
    records: Iterable[EpisodeRecord],
    *,
    dataset_root: str | Path,
    action_chunk: int = 48,
    action_dim: int = 14,
) -> tuple[dict, dict]:
    """Compute RoboNana metadata directly from FACT's LeRobot parquet files."""

    import pandas as pd

    root = Path(dataset_root).expanduser().resolve()
    records = list(records)
    state_moments = RunningMoments(action_dim)
    action_moments = RunningMoments(action_dim)
    delta_mask = ALOHA_DELTA_MASK[:action_dim]
    for record in records:
        frame = pd.read_parquet(
            record.source,
            columns=["observation.state", "action", "frame_index"],
        ).sort_values("frame_index", kind="stable")
        state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32, copy=False)[:, :action_dim]
        action = np.stack(frame["action"].to_numpy()).astype(np.float32, copy=False)[:, :action_dim]
        if state.shape[0] != record.length or action.shape[0] != record.length:
            raise RuntimeError(f"Parquet length disagrees with metadata: {record.source}")
        state_moments.update(state)
        time = np.arange(record.length, dtype=np.int64)[:, None]
        offsets = np.arange(action_chunk, dtype=np.int64)[None]
        chunks = action[np.clip(time + offsets, 0, record.length - 1)].copy()
        chunks[:, :, delta_mask] -= state[:, None, delta_mask]
        action_moments.update(chunks)

    index = {
        "schema_version": 2,
        "source_format": "lerobot-v2",
        "task_globs": list(DEFAULT_TASK_GLOBS),
        "episodes": [_episode_index_row(record, root) for record in records],
    }
    stats = {
        "schema_version": 2,
        "source_format": "lerobot-v2",
        "action_representation": "ALOHA joint delta except grippers",
        "action_chunk": int(action_chunk),
        "norm_stats": {
            "observation.state": state_moments.as_dict(),
            "action": action_moments.as_dict(),
            "value": {"min": [-1.0], "max": [2.0]},
        },
    }
    return index, stats


def write_robotwin_lerobot_metadata(
    dataset_root: str | Path,
    *,
    task_globs: tuple[str, ...] | list[str] = DEFAULT_TASK_GLOBS,
    action_chunk: int = 48,
    action_dim: int = 14,
    index_path: str | Path | None = None,
    stats_path: str | Path | None = None,
) -> tuple[Path, Path]:
    root = Path(dataset_root).expanduser().resolve()
    records = discover_lerobot_episode_records(root, task_globs)
    index, stats = compute_robotwin_lerobot_metadata(
        records,
        dataset_root=root,
        action_chunk=action_chunk,
        action_dim=action_dim,
    )
    index_output = Path(index_path).expanduser() if index_path else root / "robonana_index.json"
    stats_output = Path(stats_path).expanduser() if stats_path else root / "robonana_norm_stats.json"
    for path, payload in ((index_output, index), (stats_output, stats)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    return index_output, stats_output
