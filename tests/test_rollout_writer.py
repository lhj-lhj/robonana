from __future__ import annotations

import json
from io import BytesIO

import h5py
import numpy as np
import pytest
from PIL import Image

from robonana.data.rollout_writer import CAMERAS, RoboTwinRolloutWriter


def test_rollout_writer_saves_atomic_training_episode(tmp_path) -> None:
    initial_root = tmp_path / "initial"
    rollout_root = tmp_path / "rollouts" / "step100"
    writer = RoboTwinRolloutWriter(
        rollout_root,
        initial_dataset_root=initial_root,
        checkpoint="checkpoint-100",
        task_config="demo_clean",
    )
    for step in range(2):
        images = {
            camera: np.full((8, 12, 3), step * 20 + index, dtype=np.uint8)
            for index, camera in enumerate(CAMERAS)
        }
        writer.append(
            task_name="beat_block_hammer",
            instruction="beat the block",
            seed=100000,
            images=images,
            state=np.arange(14, dtype=np.float32) + step,
            action=np.arange(14, dtype=np.float32) + step + 0.5,
            success=False,
            terminal=step == 1,
        )
    output = writer.finish_episode()
    assert output is not None
    assert output.relative_to(rollout_root).as_posix() == (
        "beat_block_hammer/robonana_rollout/data/episode0.hdf5"
    )
    with h5py.File(output, "r") as handle:
        assert handle.attrs["success"] == np.bool_(False)
        assert handle.attrs["failure_episode"] == np.bool_(True)
        np.testing.assert_allclose(handle["joint_action/vector"][:], np.arange(14)[None] + [[0], [1]])
        np.testing.assert_allclose(handle["policy_action/vector"][:], np.arange(14)[None] + [[0.5], [1.5]])
        encoded = bytes(handle["observation/head_camera/rgb"][0])
        assert Image.open(BytesIO(encoded)).size == (12, 8)
    metadata = json.loads(
        (rollout_root / "beat_block_hammer/robonana_rollout/metadata/episode0.json").read_text()
    )
    assert metadata["failure_episode"] is True
    assert metadata["length"] == 2
    assert not writer.has_pending_episode


def test_rollout_writer_rejects_initial_dataset_subdirectory(tmp_path) -> None:
    initial_root = tmp_path / "initial"
    with pytest.raises(ValueError, match="must be separate"):
        RoboTwinRolloutWriter(initial_root / "policy_data", initial_dataset_root=initial_root)
