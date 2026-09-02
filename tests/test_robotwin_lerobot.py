from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import robonana.data.robotwin_lerobot as robotwin_lerobot  # noqa: E402
from robonana.data.robotwin_lerobot import (  # noqa: E402
    RoboTwinLeRobotDataset,
    discover_lerobot_episode_records,
    load_lerobot_episode_records,
)
from robonana.data.stats import compute_robotwin_lerobot_metadata  # noqa: E402
from world_action_model.image_layouts import ROBOTWIN_VIEW_KEYS  # noqa: E402


def _write_dataset(root):
    task = root / "Clean" / "test_task"
    (task / "meta").mkdir(parents=True)
    (task / "data" / "chunk-000").mkdir(parents=True)
    (task / "flux_cache" / "latents").mkdir(parents=True)
    (task / "flux_cache" / "language").mkdir(parents=True)
    row = {"episode_index": 0, "length": 3, "tasks": ["test the robot"]}
    (task / "meta" / "episodes.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    state = np.arange(42, dtype=np.float32).reshape(3, 14) / 10
    action = state + 1
    frame = pd.DataFrame(
        {
            "observation.state": list(state),
            "action": list(action),
            "frame_index": np.arange(3),
            "timestamp": np.asarray([0.0, 0.1, 0.2]),
        }
    )
    frame.to_parquet(task / "data" / "chunk-000" / "episode_000000.parquet")
    torch.save(torch.arange(3 * 4 * 2, dtype=torch.bfloat16).reshape(3, 4, 2), task / "flux_cache" / "latents" / "episode_000000.pt")
    torch.save(torch.ones(2, 8, dtype=torch.bfloat16), task / "flux_cache" / "language" / "episode_000000.pt")
    return task, state, action


def test_lerobot_adapter_preserves_horizon_and_tail_clip(tmp_path):
    root = tmp_path / "RoboTwin"
    _, state, action = _write_dataset(root)
    stats = {
        "norm_stats": {
            "observation.state": {"mean": [0] * 14, "std": [1] * 14},
            "action": {"mean": [0] * 14, "std": [1] * 14},
            "value": {"min": [-1], "max": [2]},
        }
    }
    stats_path = root / "robonana_norm_stats.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    dataset = RoboTwinLeRobotDataset(
        str(root),
        stats_path=str(stats_path),
        task_globs=("Clean/*",),
        action_chunk=4,
        fixed_horizon=2,
    )
    dataset.open()
    sample = dataset._get_data(0)
    assert sample["horizon_idx"].item() == 2
    assert sample["future_index"].item() == 2
    assert torch.equal(sample["current_latents"], torch.arange(8, dtype=torch.bfloat16).reshape(4, 2))
    assert torch.equal(sample["future_latents"], torch.arange(16, 24, dtype=torch.bfloat16).reshape(4, 2))
    expected = np.stack([action[0], action[1], action[2], action[2]])
    delta_mask = np.array([True] * 6 + [False] + [True] * 6 + [False])
    expected[:, delta_mask] -= state[0, delta_mask]
    np.testing.assert_allclose(sample["action"].numpy(), expected)


def test_lerobot_metadata_has_full_source_contract(tmp_path):
    root = tmp_path / "RoboTwin"
    _write_dataset(root)
    records = discover_lerobot_episode_records(root, ("Clean/*",))
    index, stats = compute_robotwin_lerobot_metadata(records, dataset_root=root, action_chunk=4)
    assert index["source_format"] == "lerobot-v2"
    assert index["episodes"][0]["source"].endswith("episode_000000.parquet")
    assert stats["source_format"] == "lerobot-v2"
    assert stats["action_chunk"] == 4


def test_global_lerobot_index_still_respects_task_globs(tmp_path):
    root = tmp_path / "RoboTwin"
    _write_dataset(root)
    randomized = root / "Randomized" / "other_task"
    (randomized / "meta").mkdir(parents=True)
    (randomized / "data" / "chunk-000").mkdir(parents=True)
    row = {"episode_index": 0, "length": 1, "tasks": ["other task"]}
    (randomized / "meta" / "episodes.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        {
            "observation.state": [np.zeros(14, dtype=np.float32)],
            "action": [np.zeros(14, dtype=np.float32)],
            "frame_index": [0],
            "timestamp": [0.0],
        }
    ).to_parquet(randomized / "data" / "chunk-000" / "episode_000000.parquet")

    all_records = discover_lerobot_episode_records(root, ("Clean/*", "Randomized/*"))
    index, _ = compute_robotwin_lerobot_metadata(
        all_records, dataset_root=root, action_chunk=4
    )
    index_path = root / "robonana_index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    records = load_lerobot_episode_records(
        root,
        ("Clean/test_task",),
        index_path,
    )
    assert len(records) == 1
    assert records[0].task_name == "test_task"
    assert records[0].task_dir == (root / "Clean" / "test_task").resolve()


def test_online_dino_decodes_only_the_horizon_selected_three_view_frame(tmp_path, monkeypatch):
    root = tmp_path / "RoboTwin"
    task, _, _ = _write_dataset(root)
    for view_key in ROBOTWIN_VIEW_KEYS:
        path = task / "videos" / "chunk-000" / view_key / "episode_000000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    stats_path = root / "robonana_norm_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "observation.state": {"mean": [0] * 14, "std": [1] * 14},
                    "action": {"mean": [0] * 14, "std": [1] * 14},
                    "value": {"min": [-1], "max": [2]},
                }
            }
        ),
        encoding="utf-8",
    )
    decoded_timestamps = []

    def fake_decode(path, timestamps, kwargs):
        decoded_timestamps.append((path, timestamps.copy(), kwargs.copy()))
        height = 8 if "cam_high" in path else 4
        return np.full((1, height, 6, 3), 17, dtype=np.uint8)

    monkeypatch.setattr(robotwin_lerobot, "_decode_frames_by_timestamps_pyav", fake_decode)
    dataset = RoboTwinLeRobotDataset(
        str(root),
        stats_path=str(stats_path),
        task_globs=("Clean/*",),
        action_chunk=4,
        fixed_horizon=2,
        dino_online=True,
    )
    dataset.open()
    sample = dataset._get_data(0)
    assert tuple(sample["future_dino_images"]) == ROBOTWIN_VIEW_KEYS
    assert [tuple(image.shape) for image in sample["future_dino_images"].values()] == [
        (3, 8, 6),
        (3, 4, 6),
        (3, 4, 6),
    ]
    assert len(decoded_timestamps) == 3
    assert all(row[1].tolist() == [0.2] for row in decoded_timestamps)

    dataset.dino_image_size = (8, 6)
    resized = dataset._get_data(0)["future_dino_images"]
    assert [tuple(image.shape) for image in resized.values()] == [
        (3, 8, 6),
        (3, 8, 6),
        (3, 8, 6),
    ]
