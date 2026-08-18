import h5py
import numpy as np

from robonana.data.robotwin_hdf5 import discover_episode_records
from robonana.data.stats import compute_robotwin_metadata


def test_metadata_uses_fact_tail_clipped_action_chunks(tmp_path):
    root = tmp_path / "dataset"
    task_dir = root / "task" / "aloha-agilex_clean_50"
    (task_dir / "data").mkdir(parents=True)
    vector = np.zeros((2, 14), dtype=np.float32)
    vector[1] = 2.0
    with h5py.File(task_dir / "data" / "episode0.hdf5", "w") as handle:
        handle.create_dataset("joint_action/vector", data=vector)

    records = discover_episode_records(root, "*/aloha-agilex_clean_50")
    index, stats = compute_robotwin_metadata(records, dataset_root=root, action_chunk=2)

    assert index["episodes"][0]["source"] == "task/aloha-agilex_clean_50/data/episode0.hdf5"
    assert index["episodes"][0]["task_dir"] == "task/aloha-agilex_clean_50"
    np.testing.assert_allclose(stats["norm_stats"]["observation.state"]["mean"], np.ones(14))
    # Delta joint dimensions see [0, 2, 0, 0], while grippers remain [0, 2, 2, 2].
    assert stats["norm_stats"]["action"]["mean"][0] == 0.5
    assert stats["norm_stats"]["action"]["mean"][6] == 1.5
