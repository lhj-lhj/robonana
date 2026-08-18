import json

import h5py
import numpy as np
import torch

from robonana.data.robotwin_hdf5 import RoboTwinEpisodeSampler, RoboTwinHDF5Dataset


def _stats(dim=14):
    zeros = [0.0] * dim
    ones = [1.0] * dim
    return {
        "norm_stats": {
            "observation.state": {"mean": zeros, "std": ones},
            "action": {"mean": zeros, "std": ones},
            "value": {"min": [-1.0], "max": [2.0]},
        }
    }


def test_hdf5_dataset_uses_fact_tail_clip_and_cached_flux_tokens(tmp_path):
    root = tmp_path / "hf_dataset"
    task_dir = root / "adjust_bottle" / "aloha-agilex_clean_50"
    (task_dir / "data").mkdir(parents=True)
    (task_dir / "flux_cache" / "latents").mkdir(parents=True)
    vectors = np.arange(5 * 14, dtype=np.float32).reshape(5, 14)
    with h5py.File(task_dir / "data" / "episode0.hdf5", "w") as handle:
        handle.create_dataset("joint_action/vector", data=vectors)
    frame_latents = torch.arange(5 * 3 * 4).reshape(5, 3, 4).to(torch.bfloat16)
    torch.save(frame_latents, task_dir / "flux_cache" / "latents" / "episode_000000.pt")
    torch.save(torch.zeros(8, 12, dtype=torch.bfloat16), task_dir / "flux_cache" / "language_context.pt")
    stats_path = root / "norm_stats.json"
    stats_path.write_text(json.dumps(_stats()), encoding="utf-8")

    dataset = RoboTwinHDF5Dataset(
        str(root),
        stats_path=str(stats_path),
        action_chunk=4,
        max_horizon=4,
        fixed_horizon=4,
        eval_horizons=(1, 2, 4),
    )
    sample = dataset[3]

    assert sample["horizon_idx"].item() == 4
    assert sample["future_index"].item() == 4
    torch.testing.assert_close(sample["current_latents"], frame_latents[3])
    torch.testing.assert_close(sample["future_latents"], frame_latents[4])
    assert sample["eval_horizon_idx"].tolist() == [1, 2, 4]
    assert sample["eval_future_index"].tolist() == [4, 4, 4]
    torch.testing.assert_close(sample["eval_future_latents"], frame_latents[4].expand(3, -1, -1))
    expected = torch.from_numpy(vectors[[3, 4, 4, 4]].copy())
    delta_mask = torch.tensor([True] * 6 + [False] + [True] * 6 + [False])
    expected[:, delta_mask] -= torch.from_numpy(vectors[3].copy())[delta_mask]
    torch.testing.assert_close(sample["action"], expected)


def test_episode_sampler_returns_valid_indices(tmp_path):
    root = tmp_path / "hf_dataset"
    task_dir = root / "task" / "aloha-agilex_clean_50"
    (task_dir / "data").mkdir(parents=True)
    (task_dir / "flux_cache" / "latents").mkdir(parents=True)
    with h5py.File(task_dir / "data" / "episode0.hdf5", "w") as handle:
        handle.create_dataset("joint_action/vector", data=np.zeros((3, 14), dtype=np.float32))
    torch.save(torch.zeros(3, 2, 4), task_dir / "flux_cache" / "latents" / "episode_000000.pt")
    torch.save(torch.zeros(2, 3), task_dir / "flux_cache" / "language_context.pt")
    stats_path = root / "norm_stats.json"
    stats_path.write_text(json.dumps(_stats()), encoding="utf-8")
    dataset = RoboTwinHDF5Dataset(
        str(root), stats_path=str(stats_path), fixed_horizon=1, eval_horizons=(1,)
    )
    sampler = RoboTwinEpisodeSampler(dataset, infinite=False, sample_epoch_size=7, seed=3)
    indices = list(iter(sampler))
    assert len(indices) == 7
    assert all(0 <= index < len(dataset) for index in indices)
