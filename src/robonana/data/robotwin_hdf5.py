"""FACT-compatible dataset adapter for the released RoboTwin HDF5 files."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import torch

from fact_datasets.datasets.base_dataset import BaseDataset
from fact_datasets.datasets.dataset import register_dataset
from fact_train.samplers.build import SAMPLERS

from .flux_cache import (
    episode_cache_path,
    episode_language_context_path,
    language_context_path,
    select_current_future_latents,
)


ALOHA_DELTA_MASK = np.asarray(
    [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
    dtype=bool,
)


@dataclass(frozen=True)
class EpisodeRecord:
    task_name: str
    task_dir: Path
    source: Path
    episode_index: int
    length: int
    success: bool = True


def _episode_index(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"Unexpected RoboTwin episode filename: {path}")
    return int(match.group(1))


def discover_episode_records(dataset_root: str | Path, task_glob: str) -> list[EpisodeRecord]:
    root = Path(dataset_root).expanduser().resolve()
    records: list[EpisodeRecord] = []
    for task_dir in sorted(root.glob(task_glob)):
        for source in sorted((task_dir / "data").glob("episode*.hdf5"), key=_episode_index):
            with h5py.File(source, "r") as handle:
                length = int(handle["joint_action/vector"].shape[0])
                success = bool(handle.attrs.get("success", True))
            if length <= 0:
                continue
            records.append(
                EpisodeRecord(
                    task_name=task_dir.parent.name,
                    task_dir=task_dir.resolve(),
                    source=source.resolve(),
                    episode_index=_episode_index(source),
                    length=length,
                    success=success,
                )
            )
    if not records:
        raise FileNotFoundError(f"No RoboTwin HDF5 episodes matching {task_glob!r} under {root}")
    return records


def load_episode_records(dataset_root: str | Path, task_glob: str, index_path: str | Path | None) -> list[EpisodeRecord]:
    root = Path(dataset_root).expanduser().resolve()
    path = Path(index_path).expanduser().resolve() if index_path else root / "robonana_index.json"
    if not path.is_file():
        return discover_episode_records(root, task_glob)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in payload["episodes"]:
        task_dir = root / row["task_dir"]
        records.append(
            EpisodeRecord(
                task_name=str(row["task_name"]),
                task_dir=task_dir,
                source=root / row["source"],
                episode_index=int(row["episode_index"]),
                length=int(row["length"]),
                success=bool(row.get("success", True)),
            )
        )
    return records


def _stats_array(stats: dict[str, Any], key: str, field: str, dim: int) -> np.ndarray:
    values = np.asarray(stats["norm_stats"][key][field], dtype=np.float32).reshape(-1)
    if values.size < dim:
        values = np.pad(values, (0, dim - values.size), constant_values=0.0 if field == "mean" else 1.0)
    return values[:dim]


@register_dataset
class RoboTwinHDF5Dataset(BaseDataset):
    """Read raw RoboTwin state/action and pair it with cached FLUX/Qwen tokens."""

    def __init__(
        self,
        data_path: str,
        *,
        stats_path: str,
        task_glob: str = "*/aloha-agilex_clean_50",
        index_path: str | None = None,
        action_chunk: int = 48,
        action_dim: int = 14,
        max_horizon: int = 48,
        fixed_horizon: int = 0,
        rollout_horizon: int | None = None,
        rollout_horizon_prob: float = 0.5,
        eval_horizons: tuple[int, ...] | list[int] = (12, 24, 48),
        latent_cache_size: int = 4,
        language_cache_size: int = 8,
        hdf5_cache_size: int = 4,
    ) -> None:
        super().__init__(data_path=data_path)
        self.stats_path = str(stats_path)
        self.task_glob = str(task_glob)
        self.index_path = index_path
        self.action_chunk = int(action_chunk)
        self.action_dim = int(action_dim)
        self.max_horizon = int(max_horizon)
        self.fixed_horizon = int(fixed_horizon)
        self.rollout_horizon = (
            max(1, self.max_horizon // 2)
            if rollout_horizon is None
            else int(rollout_horizon)
        )
        self.rollout_horizon_prob = float(rollout_horizon_prob)
        self.eval_horizons = tuple(int(value) for value in eval_horizons)
        self.latent_cache_size = int(latent_cache_size)
        self.language_cache_size = int(language_cache_size)
        self.hdf5_cache_size = int(hdf5_cache_size)
        if self.action_chunk <= 0 or self.max_horizon <= 0:
            raise ValueError("action_chunk and max_horizon must be positive")
        if self.action_dim <= 0 or self.action_dim > ALOHA_DELTA_MASK.size:
            raise ValueError(f"action_dim must lie in [1, {ALOHA_DELTA_MASK.size}]")
        if self.fixed_horizon < 0 or self.fixed_horizon > self.max_horizon:
            raise ValueError("fixed_horizon must be 0 or lie in [1, max_horizon]")
        if self.rollout_horizon < 1 or self.rollout_horizon > self.max_horizon:
            raise ValueError("rollout_horizon must lie in [1, max_horizon]")
        if not 0.0 <= self.rollout_horizon_prob <= 1.0:
            raise ValueError("rollout_horizon_prob must lie in [0, 1]")
        if not self.eval_horizons or any(value < 1 or value > self.max_horizon for value in self.eval_horizons):
            raise ValueError("eval_horizons must be non-empty and lie in [1, max_horizon]")
        if min(self.latent_cache_size, self.language_cache_size, self.hdf5_cache_size) < 1:
            raise ValueError("all cache sizes must be at least one")

        self.records: list[EpisodeRecord] = []
        self.episode_starts = np.empty((0,), dtype=np.int64)
        self.episode_stops = np.empty((0,), dtype=np.int64)
        self._stats: dict[str, Any] | None = None
        self._latent_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self._language_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self._hdf5_cache: OrderedDict[Path, Any] = OrderedDict()

    @classmethod
    def load(cls, data_or_config):
        config = dict(data_or_config)
        for key in list(config):
            if key.startswith("_") or key == "config_path":
                config.pop(key)
        return cls(**config)

    def _ensure_index(self) -> None:
        if self.records:
            return
        self.records = load_episode_records(self.data_path, self.task_glob, self.index_path)
        lengths = np.asarray([record.length for record in self.records], dtype=np.int64)
        self.episode_stops = np.cumsum(lengths)
        self.episode_starts = self.episode_stops - lengths

    def open(self) -> None:
        self._ensure_index()
        if self._stats is None:
            self._stats = json.loads(Path(self.stats_path).expanduser().read_text(encoding="utf-8"))

    def close(self) -> None:
        for handle in self._hdf5_cache.values():
            close = getattr(handle, "close", None)
            if close is not None:
                close()
        self._hdf5_cache.clear()
        self._latent_cache.clear()
        self._language_cache.clear()

    def __len__(self) -> int:
        self._ensure_index()
        return int(self.episode_stops[-1])

    @staticmethod
    def _lru_get(cache: OrderedDict, key, loader, capacity: int):
        if key in cache:
            value = cache.pop(key)
            cache[key] = value
            return value
        value = loader(key)
        cache[key] = value
        while len(cache) > capacity:
            _, evicted = cache.popitem(last=False)
            if isinstance(evicted, h5py.File):
                evicted.close()
        return value

    def _handle(self, path: Path) -> h5py.File:
        return self._lru_get(self._hdf5_cache, path, lambda p: h5py.File(p, "r"), self.hdf5_cache_size)

    def _episode_state_action(self, record: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
        """Load one episode's state and policy-action arrays.

        The LeRobot adapter overrides this method while reusing the exact same
        horizon, action-chunk, normalization, cache, and loss inputs below.
        """

        handle = self._handle(record.source)
        state = np.asarray(handle["joint_action/vector"][:, : self.action_dim], dtype=np.float32)
        action_key = "policy_action/vector" if "policy_action/vector" in handle else "joint_action/vector"
        action = np.asarray(handle[action_key][:, : self.action_dim], dtype=np.float32)
        return state, action

    def _latents(self, record: EpisodeRecord) -> torch.Tensor:
        path = episode_cache_path(record.task_dir, record.episode_index)
        return self._lru_get(
            self._latent_cache,
            path,
            lambda p: torch.load(p, map_location="cpu", weights_only=True),
            self.latent_cache_size,
        )

    def _context(self, record: EpisodeRecord) -> torch.Tensor:
        episode_path = episode_language_context_path(record.task_dir, record.episode_index)
        path = episode_path if episode_path.is_file() else language_context_path(record.task_dir)
        return self._lru_get(
            self._language_cache,
            path,
            lambda p: torch.load(p, map_location="cpu", weights_only=True),
            self.language_cache_size,
        )

    def _locate(self, index: int) -> tuple[EpisodeRecord, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_pos = int(np.searchsorted(self.episode_stops, int(index), side="right"))
        return self.records[episode_pos], int(index - self.episode_starts[episode_pos])

    def _sample_horizon(self) -> int:
        if self.fixed_horizon:
            return self.fixed_horizon
        if torch.rand(()).item() < self.rollout_horizon_prob:
            return self.rollout_horizon
        return int(torch.randint(1, self.max_horizon + 1, ()).item())

    def load_eval_future_latents(
        self,
        sample_index: int,
        horizons: tuple[int, ...] | list[int] | np.ndarray,
    ) -> torch.Tensor:
        """Load fixed-horizon GT images only when a periodic eval requests them."""

        record, frame_index = self._locate(int(sample_index))
        horizon_indices = np.asarray(horizons, dtype=np.int64).reshape(-1)
        if (
            horizon_indices.size == 0
            or np.any(horizon_indices < 1)
            or np.any(horizon_indices > self.max_horizon)
        ):
            raise ValueError("eval horizons must lie in [1, max_horizon]")
        future_indices = np.minimum(frame_index + horizon_indices, record.length - 1)
        frame_latents = self._latents(record)
        if frame_latents.shape[0] != record.length:
            raise RuntimeError(
                f"FLUX cache length {frame_latents.shape[0]} disagrees with HDF5 length "
                f"{record.length}: {record.source}"
            )
        return frame_latents[torch.from_numpy(future_indices)]

    def _get_data(self, index: int) -> dict[str, Any]:
        record, frame_index = self._locate(int(index))
        horizon_idx = self._sample_horizon()
        future_index = min(frame_index + horizon_idx, record.length - 1)
        action_indices = np.clip(
            frame_index + np.arange(self.action_chunk, dtype=np.int64),
            0,
            record.length - 1,
        )

        # Episodes are short (~140 steps). Reading the small arrays once also
        # avoids backend-specific indexed-read restrictions at clipped tails.
        vector, policy_action = self._episode_state_action(record)
        if vector.shape[0] != record.length:
            raise RuntimeError(
                f"State length {vector.shape[0]} disagrees with episode length "
                f"{record.length}: {record.source}"
            )
        if policy_action.shape[0] != record.length:
            raise RuntimeError(
                f"Policy action length {policy_action.shape[0]} disagrees with episode length "
                f"{record.length}: {record.source}"
            )
        state_raw = vector[frame_index]
        action_raw = policy_action[action_indices]
        future_state_raw = vector[future_index]

        assert self._stats is not None
        state_mean = _stats_array(self._stats, "observation.state", "mean", self.action_dim)
        state_std = _stats_array(self._stats, "observation.state", "std", self.action_dim)
        action_mean = _stats_array(self._stats, "action", "mean", self.action_dim)
        action_std = _stats_array(self._stats, "action", "std", self.action_dim)
        state_std = np.maximum(state_std, 1e-8)
        action_std = np.maximum(action_std, 1e-8)

        delta = action_raw.copy()
        delta_mask = ALOHA_DELTA_MASK[: self.action_dim]
        delta[:, delta_mask] -= state_raw[None, delta_mask]
        norm_state = (state_raw - state_mean) / state_std
        norm_future_state = (future_state_raw - state_mean) / state_std
        norm_action = (delta - action_mean) / action_std

        remaining_value = 0.0 if record.length <= 1 else (record.length - future_index - 1) / (record.length - 1)
        value_min = float(np.asarray(self._stats["norm_stats"]["value"]["min"]).reshape(-1)[0])
        value_max = float(np.asarray(self._stats["norm_stats"]["value"]["max"]).reshape(-1)[0])
        if not record.success:
            remaining_value += 1.0
        value_normalized = ((remaining_value - value_min) / max(value_max - value_min, 1e-8)) * 2.0 - 1.0
        value_normalized = float(np.clip(value_normalized, -1.0, 1.0))

        frame_latents = self._latents(record)
        if frame_latents.shape[0] != record.length:
            raise RuntimeError(
                f"FLUX cache length {frame_latents.shape[0]} disagrees with HDF5 length "
                f"{record.length}: {record.source}"
        )
        current_latent, future_latent = select_current_future_latents(frame_latents, frame_index, horizon_idx)
        context = self._context(record)
        return {
            "context": context,
            "context_mask": torch.ones(context.shape[0], dtype=torch.bool),
            "current_latents": current_latent,
            "future_latents": future_latent,
            "state": torch.from_numpy(norm_state.copy()),
            "action": torch.from_numpy(norm_action.copy()),
            "future_state": torch.from_numpy(norm_future_state.copy()),
            "value": torch.tensor([value_normalized], dtype=torch.float32),
            "horizon_idx": torch.tensor(horizon_idx, dtype=torch.long),
            "action_loss_mask": torch.tensor(float(record.success), dtype=torch.float32),
            "failure_episode_mask": torch.tensor(float(not record.success), dtype=torch.float32),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
            "future_index": torch.tensor(future_index, dtype=torch.long),
            "episode_length": torch.tensor(record.length, dtype=torch.long),
        }


@SAMPLERS.register
class RoboTwinEpisodeSampler(torch.utils.data.Sampler[int]):
    """FACT-style uniform episode sampling for the combined raw HDF5 dataset."""

    def __init__(
        self,
        dataset: RoboTwinHDF5Dataset,
        batch_size: int | None = None,
        shuffle: bool = True,
        infinite: bool = True,
        seed: int = 6666,
        sample_epoch_size: int | None = None,
    ) -> None:
        dataset._ensure_index()
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.infinite = bool(infinite)
        self.seed = int(seed)
        self.epoch = 0
        size = int(sample_epoch_size or len(dataset))
        self.total_size = size if batch_size is None else int(np.ceil(size / batch_size) * batch_size)

    def __len__(self) -> int:
        return self.total_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        while True:
            rng = np.random.default_rng(self.seed + self.epoch)
            self.epoch += 1
            episodes = rng.integers(0, len(self.dataset.records), size=self.total_size)
            lengths = self.dataset.episode_stops[episodes] - self.dataset.episode_starts[episodes]
            offsets = (rng.random(self.total_size) * lengths).astype(np.int64)
            indices = self.dataset.episode_starts[episodes] + offsets
            if self.shuffle:
                rng.shuffle(indices)
            yield from indices.tolist()
            if not self.infinite:
                return


@SAMPLERS.register
class RoboTwinMixtureSampler(torch.utils.data.Sampler[int]):
    """Episode-uniform sampling across physically separate RoboTwin datasets."""

    def __init__(
        self,
        dataset,
        batch_size: int | None = None,
        shuffle: bool = True,
        infinite: bool = True,
        seed: int = 6666,
        dataset_weights: list[float] | None = None,
        sample_epoch_size: int | None = None,
    ) -> None:
        children = getattr(dataset, "datasets", None)
        if not children:
            raise TypeError("RoboTwinMixtureSampler requires a ConcatDataset")
        self.dataset = dataset
        self.children = list(children)
        for child in self.children:
            if not isinstance(child, RoboTwinHDF5Dataset):
                raise TypeError(
                    f"RoboTwinMixtureSampler only supports RoboTwinHDF5Dataset children, got {type(child)}"
                )
            child._ensure_index()
        weights = np.asarray(
            dataset_weights if dataset_weights is not None else [1.0] * len(self.children),
            dtype=np.float64,
        )
        if weights.shape != (len(self.children),) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("dataset_weights must be non-negative with one positive value per dataset")
        self.dataset_probabilities = weights / weights.sum()
        self.offsets = np.cumsum([0] + [len(child) for child in self.children[:-1]], dtype=np.int64)
        self.shuffle = bool(shuffle)
        self.infinite = bool(infinite)
        self.seed = int(seed)
        self.epoch = 0
        size = int(sample_epoch_size or sum(len(child) for child in self.children))
        self.total_size = size if batch_size is None else int(np.ceil(size / batch_size) * batch_size)

    def __len__(self) -> int:
        return self.total_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        while True:
            rng = np.random.default_rng(self.seed + self.epoch)
            self.epoch += 1
            child_indices = rng.choice(
                len(self.children),
                size=self.total_size,
                p=self.dataset_probabilities,
            )
            indices = np.empty(self.total_size, dtype=np.int64)
            for child_index, child in enumerate(self.children):
                mask = child_indices == child_index
                count = int(mask.sum())
                if count == 0:
                    continue
                episode_positions = rng.integers(0, len(child.records), size=count)
                lengths = child.episode_stops[episode_positions] - child.episode_starts[episode_positions]
                frame_offsets = (rng.random(count) * lengths).astype(np.int64)
                indices[mask] = (
                    self.offsets[child_index]
                    + child.episode_starts[episode_positions]
                    + frame_offsets
                )
            if self.shuffle:
                rng.shuffle(indices)
            yield from indices.tolist()
            if not self.infinite:
                return
