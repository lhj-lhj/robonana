"""FACT-compatible dataset adapter for the released RoboTwin HDF5 files."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fact_datasets.datasets.base_dataset import BaseDataset
from fact_datasets.datasets.dataset import register_dataset
from fact_train.samplers.build import SAMPLERS
from world_action_model.image_layouts import ROBOTWIN_VIEW_KEYS

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

POSTTRAIN_POOL_IDS = {
    "original_success": 0,
    "collected_success_replay": 1,
    "historical_failure_replay": 2,
    "latest_failure": 3,
}


def discounted_chunk_reward(
    delta_steps: int,
    *,
    discount: float = 0.999,
    reward_non_goal: float = -1.0,
) -> float:
    """Return the real-trajectory discounted reward for ``delta_steps``."""

    delta_steps = int(delta_steps)
    if delta_steps < 0:
        raise ValueError("delta_steps cannot be negative")
    return float(
        sum(float(discount) ** offset * float(reward_non_goal) for offset in range(delta_steps))
    )


def mac_success_targets(
    *,
    frame_index: int,
    horizon_idx: int,
    episode_length: int,
    discount: float = 0.999,
    reward_non_goal: float = -1.0,
    reward_goal: float = 0.0,
) -> tuple[int, int, float, float]:
    """Return ``(future_index, delta, reward_h, q_mc)`` for a successful episode.

    ``reward_h`` covers only the clipped action prefix ``t:t+delta``. ``q_mc``
    is the complete discounted return from the current frame ``t`` and is
    therefore independent of the sampled horizon.
    """

    frame_index = int(frame_index)
    horizon_idx = int(horizon_idx)
    episode_length = int(episode_length)
    discount = float(discount)
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if not 0 <= frame_index < episode_length:
        raise ValueError("frame_index must lie inside the episode")
    if horizon_idx < 0:
        raise ValueError("horizon_idx must be non-negative")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must lie in (0, 1]")

    final_index = episode_length - 1
    future_index = min(frame_index + horizon_idx, final_index)
    delta = future_index - frame_index
    reward_h = discounted_chunk_reward(
        delta, discount=discount, reward_non_goal=reward_non_goal
    )
    remaining_non_goal = final_index - frame_index
    q_mc = sum(
        discount**offset * float(reward_non_goal)
        for offset in range(remaining_non_goal)
    )
    q_mc += discount**remaining_non_goal * float(reward_goal)
    return future_index, delta, float(reward_h), float(q_mc)


def mc_episode_q_target(
    *,
    frame_index: int,
    episode_length: int,
    success: bool,
    discount: float = 0.999,
    reward_non_goal: float = -1.0,
    reward_goal: float = 0.0,
    failure_terminal_q: float = -1000.0,
) -> float:
    """Return the full-episode MC Q target from one recorded observation.

    Successful trajectories terminate at ``reward_goal``. Failed trajectories
    use an explicit terminal continuation value so irrecoverable final states
    remain distinguishable from successful terminals.
    """

    frame_index = int(frame_index)
    episode_length = int(episode_length)
    discount = float(discount)
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if not 0 <= frame_index < episode_length:
        raise ValueError("frame_index must lie inside the episode")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must lie in (0, 1]")
    remaining = episode_length - 1 - frame_index
    q_mc = discounted_chunk_reward(
        remaining,
        discount=discount,
        reward_non_goal=reward_non_goal,
    )
    terminal_value = reward_goal if success else failure_terminal_q
    return float(q_mc + discount**remaining * float(terminal_value))


@dataclass(frozen=True)
class EpisodeRecord:
    task_name: str
    task_dir: Path
    source: Path
    episode_index: int
    length: int
    success: bool = True
    round_id: int = -1
    policy_checkpoint: str = ""
    policy_version: str = ""
    has_final_observation: bool = True
    time_limit_truncated: bool = False


def _attr_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


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
                round_id = int(handle.attrs.get("round_id", -1))
                policy_checkpoint = _attr_text(
                    handle.attrs.get("policy_checkpoint", handle.attrs.get("checkpoint", ""))
                )
                policy_version = _attr_text(handle.attrs.get("policy_version", ""))
                has_final_observation = bool(
                    handle.attrs.get("has_final_observation", success)
                )
                time_limit_truncated = bool(
                    handle.attrs.get("time_limit_truncated", not success)
                )
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
                    round_id=round_id,
                    policy_checkpoint=policy_checkpoint,
                    policy_version=policy_version,
                    has_final_observation=has_final_observation,
                    time_limit_truncated=time_limit_truncated,
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
                round_id=int(row.get("round_id", -1)),
                policy_checkpoint=str(
                    row.get("policy_checkpoint", row.get("checkpoint", ""))
                ),
                policy_version=str(row.get("policy_version", "")),
                has_final_observation=bool(
                    row.get("has_final_observation", row.get("success", True))
                ),
                time_limit_truncated=bool(
                    row.get("time_limit_truncated", not row.get("success", True))
                ),
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
        eval_horizons: tuple[int, ...] | list[int] = (12, 24, 48),
        latent_cache_size: int = 4,
        language_cache_size: int = 8,
        hdf5_cache_size: int = 4,
        dino_online: bool = False,
        dino_image_size: tuple[int, int] | list[int] | None = None,
        discount: float = 0.999,
        reward_non_goal: float = -1.0,
        reward_goal: float = 0.0,
        failure_terminal_q: float = -1000.0,
        q_target_mode: str = "mc_success",
        episode_filter: str | None = None,
        pool_name: str = "original_success",
        round_min: int | None = None,
        round_max: int | None = None,
        round_id: int | None = None,
        allow_empty: bool = False,
        require_final_observation: bool = False,
    ) -> None:
        super().__init__(data_path=data_path)
        self.stats_path = str(stats_path)
        self.task_glob = str(task_glob)
        self.index_path = index_path
        self.action_chunk = int(action_chunk)
        self.action_dim = int(action_dim)
        self.max_horizon = int(max_horizon)
        self.fixed_horizon = int(fixed_horizon)
        self.eval_horizons = tuple(int(value) for value in eval_horizons)
        self.latent_cache_size = int(latent_cache_size)
        self.language_cache_size = int(language_cache_size)
        self.hdf5_cache_size = int(hdf5_cache_size)
        self.dino_online = bool(dino_online)
        self.dino_image_size = (
            None
            if dino_image_size is None
            else tuple(int(value) for value in dino_image_size)
        )
        self.discount = float(discount)
        self.reward_non_goal = float(reward_non_goal)
        self.reward_goal = float(reward_goal)
        self.failure_terminal_q = float(failure_terminal_q)
        self.q_target_mode = str(q_target_mode)
        self.episode_filter = str(
            episode_filter
            or ("success" if self.q_target_mode == "mc_success" else "all")
        )
        self.pool_name = str(pool_name)
        self.round_min = None if round_min is None else int(round_min)
        self.round_max = None if round_max is None else int(round_max)
        self.selected_round_id = None if round_id is None else int(round_id)
        self.allow_empty = bool(allow_empty)
        self.require_final_observation = bool(require_final_observation)
        if self.action_chunk <= 0 or self.max_horizon <= 0:
            raise ValueError("action_chunk and max_horizon must be positive")
        if self.action_dim <= 0 or self.action_dim > ALOHA_DELTA_MASK.size:
            raise ValueError(f"action_dim must lie in [1, {ALOHA_DELTA_MASK.size}]")
        if self.fixed_horizon < 0 or self.fixed_horizon > self.max_horizon:
            raise ValueError("fixed_horizon must be 0 or lie in [1, max_horizon]")
        if not self.eval_horizons or any(value < 1 or value > self.max_horizon for value in self.eval_horizons):
            raise ValueError("eval_horizons must be non-empty and lie in [1, max_horizon]")
        if min(self.latent_cache_size, self.language_cache_size, self.hdf5_cache_size) < 1:
            raise ValueError("all cache sizes must be at least one")
        if self.dino_image_size is not None and (
            len(self.dino_image_size) != 2
            or any(value <= 0 for value in self.dino_image_size)
        ):
            raise ValueError("dino_image_size must be (height, width) with positive values")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must lie in (0, 1]")
        if self.q_target_mode not in {"mc_success", "mc_posttrain", "td_posttrain"}:
            raise ValueError(
                "q_target_mode must be 'mc_success', 'mc_posttrain', or 'td_posttrain'"
            )
        if self.episode_filter not in {"all", "success", "failure"}:
            raise ValueError("episode_filter must be all, success, or failure")
        if self.pool_name not in POSTTRAIN_POOL_IDS:
            raise ValueError(f"unknown posttrain pool_name: {self.pool_name}")
        if self.q_target_mode == "mc_success" and self.episode_filter != "success":
            raise ValueError("mc_success pretraining only accepts successful episodes")
        if self.selected_round_id is not None and (
            self.round_min is not None or self.round_max is not None
        ):
            raise ValueError("round_id cannot be combined with round_min/round_max")

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
        self._set_records(load_episode_records(self.data_path, self.task_glob, self.index_path))

    def _set_records(self, records: list[EpisodeRecord]) -> None:
        """Build one physical-pool view without merging source directories."""

        if self.episode_filter == "success":
            records = [record for record in records if record.success]
        elif self.episode_filter == "failure":
            records = [record for record in records if not record.success]
        if self.selected_round_id is not None:
            records = [record for record in records if record.round_id == self.selected_round_id]
        if self.round_min is not None:
            records = [record for record in records if record.round_id >= self.round_min]
        if self.round_max is not None:
            records = [record for record in records if record.round_id <= self.round_max]
        missing_final = [
            record.source for record in records if not record.has_final_observation
        ]
        if self.require_final_observation and missing_final:
            raise RuntimeError(
                "posttraining replay requires reset-pre final observations; recollect these episodes: "
                + ", ".join(str(path) for path in missing_final[:5])
            )
        self.records = records
        if not self.records:
            if self.allow_empty:
                self.episode_starts = np.empty((0,), dtype=np.int64)
                self.episode_stops = np.empty((0,), dtype=np.int64)
                return
            if self.q_target_mode == "mc_success":
                raise FileNotFoundError(
                    "q_target_mode='mc_success' requires at least one successful episode"
                )
            raise FileNotFoundError(f"posttrain pool {self.pool_name!r} contains no episodes")
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
        return 0 if not self.records else int(self.episode_stops[-1])

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

    def _episode_transition_valid(self, record: EpisodeRecord) -> np.ndarray:
        """Return which rows execute a real transition to the next observation."""

        handle = self._handle(record.source)
        if isinstance(handle, h5py.File) and "transition_valid" in handle:
            valid = np.asarray(handle["transition_valid"], dtype=bool).reshape(-1)
            if valid.shape != (record.length,):
                raise RuntimeError(
                    f"transition_valid has shape {valid.shape}, expected {(record.length,)}"
                )
            return valid
        # Released success demonstrations end on a terminal observation. Old
        # failure rollouts are rejected when require_final_observation=True.
        valid = np.ones(record.length, dtype=bool)
        valid[-1] = False
        return valid

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

    def _future_dino_images(self, record: EpisodeRecord, future_index: int) -> dict[str, torch.Tensor]:
        """Decode exactly one horizon-selected RGB frame from each HDF5 camera."""

        camera_names = ("head_camera", "left_camera", "right_camera")
        handle = self._handle(record.source)
        images = {}
        for view_key, camera_name in zip(ROBOTWIN_VIEW_KEYS, camera_names, strict=True):
            dataset_key = f"observation/{camera_name}/rgb"
            if dataset_key not in handle:
                raise KeyError(f"online DINO requires {dataset_key} in {record.source}")
            with Image.open(BytesIO(bytes(handle[dataset_key][future_index]))) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            images[view_key] = self._standardize_dino_image(
                torch.from_numpy(array).permute(2, 0, 1)
            )
        return images

    def _standardize_dino_image(self, image: torch.Tensor) -> torch.Tensor:
        """Give mixed replay pools one collatable online-DINO image shape."""

        if self.dino_image_size is None or tuple(image.shape[-2:]) == self.dino_image_size:
            return image
        resized = F.interpolate(
            image.unsqueeze(0).float(),
            size=self.dino_image_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return resized.squeeze(0).round().clamp_(0, 255).to(torch.uint8)

    def _locate(self, index: int) -> tuple[EpisodeRecord, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_pos = int(np.searchsorted(self.episode_stops, int(index), side="right"))
        return self.records[episode_pos], int(index - self.episode_starts[episode_pos])

    def _sample_horizon(self) -> int:
        if self.fixed_horizon:
            return self.fixed_horizon
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
        transition_valid = self._episode_transition_valid(record)
        delta_steps = int(transition_valid[frame_index:future_index].sum())
        reward_h = discounted_chunk_reward(
            delta_steps,
            discount=self.discount,
            reward_non_goal=self.reward_non_goal,
        )
        if self.q_target_mode == "mc_success":
            _, _, _, q_clean = mac_success_targets(
                frame_index=frame_index,
                horizon_idx=horizon_idx,
                episode_length=record.length,
                discount=self.discount,
                reward_non_goal=self.reward_non_goal,
                reward_goal=self.reward_goal,
            )
        elif self.q_target_mode == "mc_posttrain":
            q_clean = mc_episode_q_target(
                frame_index=frame_index,
                episode_length=record.length,
                success=record.success,
                discount=self.discount,
                reward_non_goal=self.reward_non_goal,
                reward_goal=self.reward_goal,
                failure_terminal_q=self.failure_terminal_q,
            )
        else:
            # The trainer replaces this placeholder with a stop-gradient EMA TD target.
            q_clean = 0.0
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
        # The state target is one frame at t_h=min(t+idx_h, episode_end), not
        # the last state of the fixed-length action chunk.
        future_state_raw = vector[future_index]

        assert self._stats is not None
        state_mean = _stats_array(self._stats, "observation.state", "mean", self.action_dim)
        state_std = _stats_array(self._stats, "observation.state", "std", self.action_dim)
        action_mean = _stats_array(self._stats, "action", "mean", self.action_dim)
        action_std = _stats_array(self._stats, "action", "std", self.action_dim)
        state_std = np.maximum(state_std, 1e-8)
        action_std = np.maximum(action_std, 1e-8)

        action_delta = action_raw.copy()
        delta_mask = ALOHA_DELTA_MASK[: self.action_dim]
        action_delta[:, delta_mask] -= state_raw[None, delta_mask]
        norm_state = (state_raw - state_mean) / state_std
        norm_future_state = (future_state_raw - state_mean) / state_std
        norm_action = (action_delta - action_mean) / action_std

        frame_latents = self._latents(record)
        if frame_latents.shape[0] != record.length:
            raise RuntimeError(
                f"FLUX cache length {frame_latents.shape[0]} disagrees with HDF5 length "
                f"{record.length}: {record.source}"
        )
        current_latent, future_latent = select_current_future_latents(frame_latents, frame_index, horizon_idx)
        context = self._context(record)
        success_terminal_h = bool(record.success and future_index == record.length - 1)
        direct_reward_h = self.reward_goal if success_terminal_h else self.reward_non_goal
        time_limit_truncated_h = bool(
            not record.success
            and record.time_limit_truncated
            and future_index == record.length - 1
        )
        task_names = sorted({item.task_name for item in self.records})
        sample = {
            "context": context,
            "context_mask": torch.ones(context.shape[0], dtype=torch.bool),
            "current_latents": current_latent,
            "future_latents": future_latent,
            "state": torch.from_numpy(norm_state.copy()),
            "action": torch.from_numpy(norm_action.copy()),
            "behavior_action": torch.from_numpy(norm_action.copy()),
            "future_state": torch.from_numpy(norm_future_state.copy()),
            # The model reward head predicts the one-step reward attached to
            # the selected future state.  Keep reward_h separately because TD
            # posttraining needs the real discounted reward accumulated from
            # t through the clipped horizon.
            "reward": torch.tensor([direct_reward_h], dtype=torch.float32),
            "reward_h": torch.tensor([reward_h], dtype=torch.float32),
            "success": torch.tensor([float(success_terminal_h)], dtype=torch.float32),
            "q": torch.tensor([q_clean], dtype=torch.float32),
            "horizon_idx": torch.tensor(horizon_idx, dtype=torch.long),
            "delta": torch.tensor(delta_steps, dtype=torch.long),
            "delta_steps": torch.tensor(delta_steps, dtype=torch.long),
            "terminal_h": torch.tensor(float(success_terminal_h), dtype=torch.float32),
            "success_terminal_h": torch.tensor(float(success_terminal_h), dtype=torch.float32),
            "time_limit_truncated_h": torch.tensor(
                float(time_limit_truncated_h), dtype=torch.float32
            ),
            "episode_success": torch.tensor(float(record.success), dtype=torch.float32),
            "action_loss_mask": torch.tensor(
                1.0 if self.q_target_mode == "td_posttrain" else float(record.success),
                dtype=torch.float32,
            ),
            "q_loss_mask": torch.tensor(
                1.0 if self.q_target_mode == "mc_posttrain" else float(delta_steps > 0),
                dtype=torch.float32,
            ),
            "failure_episode_mask": torch.tensor(float(not record.success), dtype=torch.float32),
            "pool_id": torch.tensor(POSTTRAIN_POOL_IDS[self.pool_name], dtype=torch.long),
            "task_id": torch.tensor(task_names.index(record.task_name), dtype=torch.long),
            "episode_id": torch.tensor(record.episode_index, dtype=torch.long),
            "round_id": torch.tensor(record.round_id, dtype=torch.long),
            "policy_version": record.policy_version,
            "policy_checkpoint": record.policy_checkpoint,
            "observation_id": (
                f"{record.task_name}/episode{record.episode_index}/frame{frame_index}"
            ),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
            "future_index": torch.tensor(future_index, dtype=torch.long),
            "episode_length": torch.tensor(record.length, dtype=torch.long),
        }
        if self.dino_online:
            sample["future_dino_images"] = self._future_dino_images(record, future_index)
        return sample


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


@SAMPLERS.register
class RoboTwinPosttrainSampler(torch.utils.data.Sampler[int]):
    """Four-pool sampler: pool -> uniform task -> uniform episode -> frame."""

    POOL_ORDER = tuple(POSTTRAIN_POOL_IDS)

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        infinite: bool = True,
        seed: int = 6666,
        sample_epoch_size: int | None = None,
        pool_weights: dict[str, float] | None = None,
        redistribute_empty_historical_failure_to_latest: bool = True,
        redistribute_empty_collected_success_to_original: bool = True,
    ) -> None:
        children = getattr(dataset, "datasets", None)
        if children is None or len(children) != 4:
            raise TypeError("RoboTwinPosttrainSampler requires four ConcatDataset children")
        self.dataset = dataset
        self.children = list(children)
        for child in self.children:
            if not isinstance(child, RoboTwinHDF5Dataset):
                raise TypeError("all posttrain pools must reuse RoboTwinHDF5Dataset")
            child._ensure_index()
        names = tuple(child.pool_name for child in self.children)
        if names != self.POOL_ORDER:
            raise ValueError(f"posttrain pool order must be {self.POOL_ORDER}, got {names}")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        configured = pool_weights or {name: 0.25 for name in self.POOL_ORDER}
        if set(configured) != set(self.POOL_ORDER):
            raise ValueError(f"pool_weights must define exactly {self.POOL_ORDER}")
        weights = np.asarray([float(configured[name]) for name in self.POOL_ORDER])
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("posttrain pool weights must be non-negative and sum to one")
        sizes = np.asarray([len(child.records) for child in self.children])
        if sizes[0] == 0:
            raise ValueError("D_original_success cannot be empty")
        if sizes[1] == 0 and redistribute_empty_collected_success_to_original:
            weights[0] += weights[1]
            weights[1] = 0.0
        if sizes[2] == 0 and redistribute_empty_historical_failure_to_latest:
            weights[3] += weights[2]
            weights[2] = 0.0
        nonempty_weighted = (sizes > 0) | np.isclose(weights, 0.0)
        if not bool(nonempty_weighted.all()):
            missing = [self.POOL_ORDER[index] for index in np.where(~nonempty_weighted)[0]]
            raise ValueError(f"weighted posttrain pools are empty: {missing}")
        self.pool_probabilities = weights / weights.sum()
        self.offsets = np.cumsum(
            [0] + [len(child) for child in self.children[:-1]], dtype=np.int64
        )
        self.task_episode_positions: list[dict[str, np.ndarray]] = []
        for child in self.children:
            grouped: dict[str, list[int]] = {}
            for episode_position, record in enumerate(child.records):
                grouped.setdefault(record.task_name, []).append(episode_position)
            self.task_episode_positions.append(
                {
                    task: np.asarray(positions, dtype=np.int64)
                    for task, positions in sorted(grouped.items())
                }
            )
        self.shuffle = bool(shuffle)
        self.infinite = bool(infinite)
        self.seed = int(seed)
        self.epoch = 0
        size = int(sample_epoch_size or max(sum(len(child) for child in self.children), self.batch_size))
        self.total_size = int(np.ceil(size / self.batch_size) * self.batch_size)

    def __len__(self) -> int:
        return self.total_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batch_pool_counts(self) -> np.ndarray:
        exact = self.pool_probabilities * self.batch_size
        counts = np.floor(exact).astype(np.int64)
        remainder = self.batch_size - int(counts.sum())
        if remainder:
            order = np.argsort(-(exact - counts), kind="stable")
            counts[order[:remainder]] += 1
        return counts

    def _sample_pool(self, rng: np.random.Generator, pool_index: int, count: int) -> list[int]:
        if count == 0:
            return []
        child = self.children[pool_index]
        grouped = self.task_episode_positions[pool_index]
        tasks = tuple(grouped)
        selected = []
        for _ in range(count):
            task = tasks[int(rng.integers(0, len(tasks)))]
            positions = grouped[task]
            episode_position = int(positions[int(rng.integers(0, len(positions)))])
            length = int(
                child.episode_stops[episode_position] - child.episode_starts[episode_position]
            )
            frame_offset = int(rng.integers(0, length))
            selected.append(
                int(self.offsets[pool_index] + child.episode_starts[episode_position] + frame_offset)
            )
        return selected

    def __iter__(self) -> Iterator[int]:
        counts = self._batch_pool_counts()
        while True:
            rng = np.random.default_rng(self.seed + self.epoch)
            self.epoch += 1
            for _ in range(self.total_size // self.batch_size):
                batch = []
                for pool_index, count in enumerate(counts.tolist()):
                    batch.extend(self._sample_pool(rng, pool_index, count))
                if self.shuffle:
                    rng.shuffle(batch)
                yield from batch
            if not self.infinite:
                return
