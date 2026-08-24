"""FACT RoboTwin-v2 LeRobot adapter for cached FLUX/Qwen inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from fact_datasets.datasets.lerobot_dataset import _decode_frames_by_timestamps_pyav
from fact_datasets.datasets.dataset import register_dataset
from world_action_model.image_layouts import ROBOTWIN_VIEW_KEYS

from .robotwin_hdf5 import EpisodeRecord, RoboTwinHDF5Dataset


DEFAULT_TASK_GLOBS = ("Clean/*", "Randomized/*")


def _task_dirs(root: Path, task_globs: Iterable[str]) -> list[Path]:
    return sorted(
        {
            path.resolve()
            for pattern in task_globs
            for path in root.glob(str(pattern))
            if (path / "meta" / "episodes.jsonl").is_file()
        }
    )


def _episode_rows(task_dir: Path) -> list[dict[str, Any]]:
    rows = []
    with (task_dir / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return sorted(rows, key=lambda row: int(row["episode_index"]))


def _parquet_path(task_dir: Path, episode_index: int) -> Path:
    direct = task_dir / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
    if direct.is_file():
        return direct.resolve()
    matches = list((task_dir / "data").glob(f"*/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one parquet for episode {episode_index} under {task_dir}, got {matches}"
        )
    return matches[0].resolve()


def _video_path(task_dir: Path, episode_index: int, view_key: str) -> Path:
    direct = (
        task_dir
        / "videos"
        / f"chunk-{episode_index // 1000:03d}"
        / view_key
        / f"episode_{episode_index:06d}.mp4"
    )
    if direct.is_file():
        return direct.resolve()
    matches = list((task_dir / "videos").glob(f"*/{view_key}/episode_{episode_index:06d}.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {view_key} video for episode {episode_index} under {task_dir}, got {matches}"
        )
    return matches[0].resolve()


def discover_lerobot_episode_records(
    dataset_root: str | Path,
    task_globs: Iterable[str] = DEFAULT_TASK_GLOBS,
) -> list[EpisodeRecord]:
    root = Path(dataset_root).expanduser().resolve()
    records = []
    for task_dir in _task_dirs(root, task_globs):
        for row in _episode_rows(task_dir):
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            if length <= 0:
                continue
            records.append(
                EpisodeRecord(
                    task_name=task_dir.name,
                    task_dir=task_dir,
                    source=_parquet_path(task_dir, episode_index),
                    episode_index=episode_index,
                    length=length,
                    success=True,
                )
            )
    if not records:
        raise FileNotFoundError(
            f"No RoboTwin LeRobot episodes matching {tuple(task_globs)!r} under {root}"
        )
    return records


def load_lerobot_episode_records(
    dataset_root: str | Path,
    task_globs: Iterable[str],
    index_path: str | Path | None,
) -> list[EpisodeRecord]:
    root = Path(dataset_root).expanduser().resolve()
    path = Path(index_path).expanduser().resolve() if index_path else root / "robonana_index.json"
    if not path.is_file():
        return discover_lerobot_episode_records(root, task_globs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_format") != "lerobot-v2":
        return discover_lerobot_episode_records(root, task_globs)
    return [
        EpisodeRecord(
            task_name=str(row["task_name"]),
            task_dir=(root / row["task_dir"]).resolve(),
            source=(root / row["source"]).resolve(),
            episode_index=int(row["episode_index"]),
            length=int(row["length"]),
            success=bool(row.get("success", True)),
        )
        for row in payload["episodes"]
    ]


def lerobot_episode_instruction(task_dir: str | Path, episode_index: int) -> str:
    for row in _episode_rows(Path(task_dir)):
        if int(row["episode_index"]) != int(episode_index):
            continue
        tasks = row.get("tasks", [])
        if isinstance(tasks, str) and tasks.strip():
            return tasks.strip()
        if isinstance(tasks, list):
            for task in tasks:
                if isinstance(task, str) and task.strip():
                    return task.strip()
        break
    return Path(task_dir).name.replace("_", " ")


@register_dataset
class RoboTwinLeRobotDataset(RoboTwinHDF5Dataset):
    """Read parquet state/action while retaining RoboNana's HDF5 data contract."""

    def __init__(
        self,
        data_path: str,
        *,
        task_globs: tuple[str, ...] | list[str] = DEFAULT_TASK_GLOBS,
        **kwargs,
    ) -> None:
        self.task_globs = tuple(str(value) for value in task_globs)
        super().__init__(data_path, task_glob=self.task_globs[0], **kwargs)

    def _ensure_index(self) -> None:
        if self.records:
            return
        self.records = load_lerobot_episode_records(self.data_path, self.task_globs, self.index_path)
        lengths = np.asarray([record.length for record in self.records], dtype=np.int64)
        self.episode_stops = np.cumsum(lengths)
        self.episode_starts = self.episode_stops - lengths

    def _episode_state_action(self, record: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
        def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
            import pandas as pd

            frame = pd.read_parquet(
                path,
                columns=["observation.state", "action", "frame_index"],
            ).sort_values("frame_index", kind="stable")
            state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32, copy=False)
            action = np.stack(frame["action"].to_numpy()).astype(np.float32, copy=False)
            return state[:, : self.action_dim], action[:, : self.action_dim]

        return self._lru_get(self._hdf5_cache, record.source, load, self.hdf5_cache_size)

    def _episode_timestamps(self, record: EpisodeRecord) -> np.ndarray:
        cache_key = (record.source, "timestamps")

        def load(_: tuple[Path, str]) -> np.ndarray:
            import pandas as pd

            try:
                frame = pd.read_parquet(
                    record.source,
                    columns=["frame_index", "timestamp"],
                ).sort_values("frame_index", kind="stable")
            except Exception as error:
                raise RuntimeError(
                    f"online DINO requires the recorded timestamp column in {record.source}"
                ) from error
            timestamps = frame["timestamp"].to_numpy(dtype=np.float64, copy=True)
            if timestamps.shape != (record.length,) or not np.isfinite(timestamps).all():
                raise RuntimeError(
                    f"invalid timestamps {timestamps.shape} for episode length {record.length}: {record.source}"
                )
            return timestamps

        return self._lru_get(self._hdf5_cache, cache_key, load, self.hdf5_cache_size)

    def _future_dino_images(self, record: EpisodeRecord, future_index: int) -> dict[str, torch.Tensor]:
        timestamp = self._episode_timestamps(record)[future_index]
        images = {}
        for view_key in ROBOTWIN_VIEW_KEYS:
            video_path = _video_path(record.task_dir, record.episode_index, view_key)
            decoded = _decode_frames_by_timestamps_pyav(
                str(video_path),
                np.asarray([timestamp], dtype=np.float64),
                {"pyav_thread_count": 1},
            )
            if decoded.shape[0] != 1 or decoded.ndim != 4 or decoded.shape[-1] != 3:
                raise RuntimeError(f"unexpected decoded frame shape {decoded.shape}: {video_path}")
            images[view_key] = torch.from_numpy(decoded[0].copy()).permute(2, 0, 1)
        return images
