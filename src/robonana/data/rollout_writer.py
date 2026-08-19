"""Atomic writer for policy-generated RoboTwin rollout collections."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from PIL import Image


CAMERAS = ("head_camera", "left_camera", "right_camera")
ROLLOUT_VARIANT = "robonana_rollout"
ROLLOUT_SCHEMA_VERSION = 1


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3:
        raise ValueError(f"RGB image must have three dimensions, got {value.shape}")
    if value.shape[-1] != 3 and value.shape[0] == 3:
        value = np.transpose(value, (1, 2, 0))
    if value.shape[-1] != 3:
        raise ValueError(f"RGB image must end in three channels, got {value.shape}")
    if value.dtype == np.uint8:
        return np.ascontiguousarray(value)
    value = value.astype(np.float32, copy=False)
    if value.max(initial=0.0) <= 1.0:
        value = value * 255.0
    return np.clip(value, 0.0, 255.0).astype(np.uint8)


def _jpeg_bytes(image: np.ndarray, quality: int) -> bytes:
    buffer = BytesIO()
    Image.fromarray(_as_uint8_rgb(image), mode="RGB").save(
        buffer,
        format="JPEG",
        quality=int(quality),
        subsampling=0,
    )
    return buffer.getvalue()


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


class RoboTwinRolloutWriter:
    """Buffer one episode as JPEG frames and publish it atomically as HDF5."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        initial_dataset_root: str | Path | None = "/workspace/datasets/RoboTwin/hf_dataset",
        variant: str = ROLLOUT_VARIANT,
        jpeg_quality: int = 95,
        policy_name: str = "robonana",
        checkpoint: str = "",
        task_config: str = "",
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if initial_dataset_root:
            initial_root = Path(initial_dataset_root).expanduser().resolve()
            if _is_within(self.dataset_root, initial_root):
                raise ValueError(
                    f"rollout dataset root must be separate from initial data: {self.dataset_root}"
                )
        self.variant = str(variant)
        self.jpeg_quality = int(jpeg_quality)
        self.policy_name = str(policy_name)
        self.checkpoint = str(checkpoint)
        self.task_config = str(task_config)
        self._frames: dict[str, list[bytes]] = {camera: [] for camera in CAMERAS}
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._metadata: dict[str, Any] | None = None
        self._success = False
        self._terminal = False
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            {
                "schema_version": ROLLOUT_SCHEMA_VERSION,
                "source": "RoboTwin policy rollout",
                "task_glob": f"*/{self.variant}",
                "initial_dataset_root": None if initial_dataset_root is None else str(initial_dataset_root),
            },
            self.dataset_root / "robonana_collection.json",
        )

    @property
    def has_pending_episode(self) -> bool:
        return bool(self._states)

    def append(
        self,
        *,
        task_name: str,
        instruction: str,
        seed: int | None,
        images: Mapping[str, np.ndarray],
        state: np.ndarray,
        action: np.ndarray,
        success: bool,
        terminal: bool,
    ) -> None:
        missing = [camera for camera in CAMERAS if camera not in images]
        if missing:
            raise KeyError(f"rollout observation is missing cameras: {missing}")
        state_value = np.asarray(state, dtype=np.float32).reshape(-1)
        action_value = np.asarray(action, dtype=np.float32).reshape(-1)
        if state_value.shape != action_value.shape:
            raise ValueError(
                f"state/action dimensions disagree: {state_value.shape} != {action_value.shape}"
            )
        metadata = {
            "task_name": str(task_name),
            "instruction": str(instruction),
            "seed": None if seed is None else int(seed),
        }
        if self._metadata is None:
            self._metadata = metadata
        elif metadata != self._metadata:
            raise RuntimeError(f"episode identity changed before finalization: {self._metadata} -> {metadata}")
        for camera in CAMERAS:
            self._frames[camera].append(_jpeg_bytes(images[camera], self.jpeg_quality))
        self._states.append(state_value.copy())
        self._actions.append(action_value.copy())
        self._success = self._success or bool(success)
        self._terminal = self._terminal or bool(terminal)

    def _reserve_episode_index(self, task_dir: Path) -> tuple[int, Path]:
        lock_dir = task_dir / ".episode_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        index = 0
        while True:
            output = task_dir / "data" / f"episode{index}.hdf5"
            lock = lock_dir / f"episode{index}.lock"
            if output.exists():
                index += 1
                continue
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                index += 1
                continue
            os.close(descriptor)
            return index, lock

    def finish_episode(self, *, force: bool = False) -> Path | None:
        if not self.has_pending_episode:
            return None
        if not self._terminal and not force:
            return None
        assert self._metadata is not None
        task_dir = self.dataset_root / self._metadata["task_name"] / self.variant
        (task_dir / "data").mkdir(parents=True, exist_ok=True)
        episode_index, lock = self._reserve_episode_index(task_dir)
        output = task_dir / "data" / f"episode{episode_index}.hdf5"
        temporary = output.with_suffix(".hdf5.tmp")
        states = np.stack(self._states).astype(np.float32, copy=False)
        actions = np.stack(self._actions).astype(np.float32, copy=False)
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            with h5py.File(temporary, "w") as handle:
                handle.attrs.update(
                    {
                        "schema_version": ROLLOUT_SCHEMA_VERSION,
                        "source": "robonana_policy_rollout",
                        "success": bool(self._success),
                        "failure_episode": not bool(self._success),
                        "task_name": self._metadata["task_name"],
                        "instruction": self._metadata["instruction"],
                        "seed": -1 if self._metadata["seed"] is None else self._metadata["seed"],
                        "policy_name": self.policy_name,
                        "checkpoint": self.checkpoint,
                        "task_config": self.task_config,
                        "created_at": created_at,
                    }
                )
                handle.create_dataset("joint_action/vector", data=states)
                handle.create_dataset("policy_action/vector", data=actions)
                image_dtype = h5py.vlen_dtype(np.dtype("uint8"))
                for camera in CAMERAS:
                    dataset = handle.create_dataset(
                        f"observation/{camera}/rgb",
                        shape=(len(self._frames[camera]),),
                        dtype=image_dtype,
                    )
                    for frame_index, encoded in enumerate(self._frames[camera]):
                        dataset[frame_index] = np.frombuffer(encoded, dtype=np.uint8)
            instruction_payload = {
                "seen": [self._metadata["instruction"]],
                "unseen": [self._metadata["instruction"]],
            }
            _atomic_json(
                instruction_payload,
                task_dir / "instructions" / f"episode{episode_index}.json",
            )
            _atomic_json(
                {
                    "schema_version": ROLLOUT_SCHEMA_VERSION,
                    "episode_index": episode_index,
                    "length": len(states),
                    "success": bool(self._success),
                    "failure_episode": not bool(self._success),
                    **self._metadata,
                    "policy_name": self.policy_name,
                    "checkpoint": self.checkpoint,
                    "task_config": self.task_config,
                    "created_at": created_at,
                    "source": str(output.relative_to(self.dataset_root)),
                },
                task_dir / "metadata" / f"episode{episode_index}.json",
            )
            # Publish the HDF5 last: dataset discovery can never see an episode
            # whose instruction or metadata sidecars are only partially written.
            temporary.replace(output)
            return output
        finally:
            if temporary.exists():
                temporary.unlink()
            if lock.exists():
                lock.unlink()
            self.reset()

    def reset(self) -> None:
        self._frames = {camera: [] for camera in CAMERAS}
        self._states = []
        self._actions = []
        self._metadata = None
        self._success = False
        self._terminal = False
