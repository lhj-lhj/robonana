import h5py
import numpy as np
import json

from robonana.data.stats import write_robotwin_replay_index


def test_replay_index_does_not_create_or_overwrite_statistics(tmp_path):
    root = tmp_path / "dataset"
    task_dir = root / "task" / "aloha-agilex_clean_50"
    (task_dir / "data").mkdir(parents=True)
    vector = np.zeros((2, 14), dtype=np.float32)
    vector[1] = 2.0
    with h5py.File(task_dir / "data" / "episode0.hdf5", "w") as handle:
        handle.create_dataset("joint_action/vector", data=vector)

    stats_path = root / "robonana_norm_stats.json"
    original = b"historical statistics must remain untouched"
    stats_path.write_bytes(original)
    output = write_robotwin_replay_index(root, task_glob="*/aloha-agilex_clean_50")
    index = json.loads(output.read_text())

    assert index["episodes"][0]["source"] == "task/aloha-agilex_clean_50/data/episode0.hdf5"
    assert index["episodes"][0]["task_dir"] == "task/aloha-agilex_clean_50"
    assert stats_path.read_bytes() == original


def test_replay_index_preserves_failure_and_terminal_metadata(tmp_path):
    root = tmp_path / "rollouts"
    task_dir = root / "task" / "robonana_rollout"
    (task_dir / "data").mkdir(parents=True)
    with h5py.File(task_dir / "data" / "episode0.hdf5", "w") as handle:
        handle.attrs["success"] = False
        handle.attrs["round_id"] = 3
        handle.attrs["policy_checkpoint"] = "checkpoint-120000"
        handle.attrs["policy_version"] = "round-3-policy"
        handle.attrs["has_final_observation"] = True
        handle.attrs["time_limit_truncated"] = True
        handle.create_dataset("joint_action/vector", data=np.zeros((2, 14), dtype=np.float32))
        handle.create_dataset("policy_action/vector", data=np.full((2, 14), 3.0, dtype=np.float32))

    index = json.loads(write_robotwin_replay_index(root).read_text())

    row = index["episodes"][0]
    assert row["failure_episode"] is True
    assert row["round_id"] == 3
    assert row["policy_checkpoint"] == "checkpoint-120000"
    assert row["policy_version"] == "round-3-policy"
    assert row["has_final_observation"] is True
    assert row["time_limit_truncated"] is True
    assert not (root / "robonana_norm_stats.json").exists()
