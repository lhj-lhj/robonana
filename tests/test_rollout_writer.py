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
        policy_version="policy-v2",
        round_id=3,
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
    final_images = {
        camera: np.full((8, 12, 3), 100 + index, dtype=np.uint8)
        for index, camera in enumerate(CAMERAS)
    }
    writer.append_final_observation(
        images=final_images,
        state=np.arange(14, dtype=np.float32) + 2,
    )
    output = writer.finish_episode()
    assert output is not None
    assert output.relative_to(rollout_root).as_posix() == (
        "beat_block_hammer/robonana_rollout/data/episode0.hdf5"
    )
    with h5py.File(output, "r") as handle:
        assert handle.attrs["success"] == np.bool_(False)
        assert handle.attrs["failure_episode"] == np.bool_(True)
        np.testing.assert_allclose(
            handle["joint_action/vector"][:], np.arange(14)[None] + [[0], [1], [2]]
        )
        np.testing.assert_allclose(
            handle["policy_action/vector"][:], np.arange(14)[None] + [[0.5], [1.5], [1.5]]
        )
        np.testing.assert_array_equal(handle["transition_valid"][:], [True, True, False])
        assert handle.attrs["has_final_observation"] == np.bool_(True)
        assert handle.attrs["time_limit_truncated"] == np.bool_(True)
        assert handle.attrs["round_id"] == 3
        assert handle.attrs["policy_version"] == "policy-v2"
        encoded = bytes(handle["observation/head_camera/rgb"][0])
        assert Image.open(BytesIO(encoded)).size == (12, 8)
    metadata = json.loads(
        (rollout_root / "beat_block_hammer/robonana_rollout/metadata/episode0.json").read_text()
    )
    assert metadata["failure_episode"] is True
    assert metadata["length"] == 3
    assert metadata["round_id"] == 3
    assert not writer.has_pending_episode


def test_rollout_writer_rejects_initial_dataset_subdirectory(tmp_path) -> None:
    initial_root = tmp_path / "initial"
    with pytest.raises(ValueError, match="must be separate"):
        RoboTwinRolloutWriter(initial_root / "policy_data", initial_dataset_root=initial_root)


def test_rollout_writer_records_q_rejection_selection(tmp_path) -> None:
    writer = RoboTwinRolloutWriter(
        tmp_path / "rollouts", initial_dataset_root=tmp_path / "initial"
    )
    images = {camera: np.zeros((4, 4, 3), dtype=np.uint8) for camera in CAMERAS}
    selection = {
        "inference_mode": "action_q_rejection",
        "candidate_q": np.asarray([-4.0, -2.0, -3.0], dtype=np.float32),
        "candidate_count": 3,
        "selected_candidate_index": 1,
        "selected_q": -2.0,
        "q_margin": 1.0,
    }
    writer.append(
        task_name="hanging_mug",
        instruction="hang the mug",
        seed=1,
        images=images,
        state=np.zeros(14, dtype=np.float32),
        action=np.zeros(14, dtype=np.float32),
        success=True,
        terminal=True,
        policy_selection=selection,
    )
    writer.append_final_observation(images=images, state=np.zeros(14, dtype=np.float32))
    output = writer.finish_episode()
    with h5py.File(output, "r") as handle:
        group = handle["policy_selection"]
        assert group.attrs["inference_mode"] == "action_q_rejection"
        np.testing.assert_array_equal(group["selected_candidate_index"][:], [1, 1])
        np.testing.assert_allclose(group["candidate_q"][:], [[-4, -2, -3], [-4, -2, -3]])
