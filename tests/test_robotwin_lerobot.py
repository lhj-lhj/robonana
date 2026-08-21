from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from robonana.data.robotwin_lerobot import (  # noqa: E402
    RoboTwinLeRobotDataset,
    discover_lerobot_episode_records,
)
from robonana.data.stats import compute_robotwin_lerobot_metadata  # noqa: E402


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
