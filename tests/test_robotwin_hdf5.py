import json

import h5py
import numpy as np
import pytest
import torch

from fact_datasets.datasets import ConcatDataset
from robonana.data.robotwin_hdf5 import (
    RoboTwinEpisodeSampler,
    RoboTwinHDF5Dataset,
    RoboTwinMixtureSampler,
    mac_success_targets,
)


def _stats(dim=14):
    zeros = [0.0] * dim
    ones = [1.0] * dim
    return {
        "norm_stats": {
            "observation.state": {"mean": zeros, "std": ones},
            "action": {"mean": zeros, "std": ones},
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
    assert sample["sample_index"].item() == 3
    torch.testing.assert_close(sample["current_latents"], frame_latents[3])
    torch.testing.assert_close(sample["future_latents"], frame_latents[4])
    assert not any(key.startswith("eval_") for key in sample)
    eval_future = dataset.load_eval_future_latents(3, (1, 2, 4))
    torch.testing.assert_close(eval_future, frame_latents[4].expand(3, -1, -1))
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


def test_horizon_sampler_is_uniform_unless_fixed(monkeypatch):
    dataset = object.__new__(RoboTwinHDF5Dataset)
    dataset._hdf5_cache = {}
    dataset._latent_cache = {}
    dataset._language_cache = {}
    dataset.fixed_horizon = 0
    dataset.max_horizon = 48

    def fake_randint(low, high, size):
        assert (low, high, size) == (1, 49, ())
        return torch.tensor(37)

    monkeypatch.setattr(torch, "randint", fake_randint)
    assert dataset._sample_horizon() == 37
    dataset.fixed_horizon = 12
    assert dataset._sample_horizon() == 12


def _write_minimal_dataset(root, task, variant, *, success=True, policy_value=None):
    task_dir = root / task / variant
    (task_dir / "data").mkdir(parents=True)
    (task_dir / "flux_cache" / "latents").mkdir(parents=True)
    (task_dir / "flux_cache" / "language").mkdir(parents=True)
    states = np.zeros((3, 14), dtype=np.float32)
    with h5py.File(task_dir / "data" / "episode0.hdf5", "w") as handle:
        handle.attrs["success"] = success
        handle.create_dataset("joint_action/vector", data=states)
        if policy_value is not None:
            handle.create_dataset(
                "policy_action/vector",
                data=np.full((3, 14), policy_value, dtype=np.float32),
            )
    torch.save(torch.zeros(3, 2, 4), task_dir / "flux_cache/latents/episode_000000.pt")
    torch.save(torch.zeros(2, 3), task_dir / "flux_cache/language/episode_000000.pt")
    stats_path = root / "norm_stats.json"
    stats_path.write_text(json.dumps(_stats()), encoding="utf-8")
    return RoboTwinHDF5Dataset(
        str(root),
        stats_path=str(stats_path),
        task_glob=f"*/{variant}",
        fixed_horizon=1,
        eval_horizons=(1,),
    )


def test_mc_success_pretraining_rejects_failure_only_dataset(tmp_path):
    dataset = _write_minimal_dataset(
        tmp_path / "rollout",
        "task",
        "robonana_rollout",
        success=False,
        policy_value=2.0,
    )
    with pytest.raises(FileNotFoundError, match="successful episode"):
        len(dataset)


def test_mixture_sampler_keeps_roots_separate_and_respects_weights(tmp_path):
    initial = _write_minimal_dataset(tmp_path / "initial", "task", "aloha-agilex_clean_50")
    rollout = _write_minimal_dataset(tmp_path / "rollout", "task", "robonana_rollout")
    combined = ConcatDataset([initial, rollout])
    sampler = RoboTwinMixtureSampler(
        combined,
        infinite=False,
        sample_epoch_size=8,
        dataset_weights=[0.0, 1.0],
        seed=4,
    )
    indices = list(sampler)
    assert len(indices) == 8
    assert all(index >= len(initial) for index in indices)


def test_future_state_reward_and_mc_q_follow_mac_targets(tmp_path):
    root = tmp_path / "hf_dataset"
    task_dir = root / "task" / "aloha-agilex_clean_50"
    (task_dir / "data").mkdir(parents=True)
    (task_dir / "flux_cache" / "latents").mkdir(parents=True)
    vectors = np.arange(5 * 14, dtype=np.float32).reshape(5, 14)
    with h5py.File(task_dir / "data" / "episode0.hdf5", "w") as handle:
        handle.create_dataset("joint_action/vector", data=vectors)
    torch.save(torch.zeros(5, 2, 4), task_dir / "flux_cache/latents/episode_000000.pt")
    torch.save(torch.zeros(2, 3), task_dir / "flux_cache/language_context.pt")
    stats_path = root / "norm_stats.json"
    stats_path.write_text(json.dumps(_stats()), encoding="utf-8")
    samples = []
    for horizon in (1, 3):
        dataset = RoboTwinHDF5Dataset(
            str(root),
            stats_path=str(stats_path),
            action_chunk=4,
            max_horizon=4,
            fixed_horizon=horizon,
            eval_horizons=(1,),
        )
        dataset.open()
        samples.append(dataset._get_data(0))

    assert [sample["future_index"].item() for sample in samples] == [1, 3]
    torch.testing.assert_close(samples[0]["future_state"], torch.from_numpy(vectors[1]))
    torch.testing.assert_close(samples[1]["future_state"], torch.from_numpy(vectors[3]))
    torch.testing.assert_close(samples[0]["reward"], torch.tensor([-1.0]))
    torch.testing.assert_close(samples[1]["reward"], torch.tensor([-2.997001]))
    expected_q = -sum(0.999**offset for offset in range(4))
    torch.testing.assert_close(samples[0]["q"], torch.tensor([expected_q]))
    torch.testing.assert_close(samples[1]["q"], torch.tensor([expected_q]))
    assert samples[0]["delta"].item() == 1
    assert samples[1]["delta"].item() == 3


def test_mac_targets_clip_tail_and_terminal_has_zero_reward_and_q():
    future_index, delta, reward, q = mac_success_targets(
        frame_index=3,
        horizon_idx=48,
        episode_length=5,
    )
    assert (future_index, delta) == (4, 1)
    assert reward == pytest.approx(-1.0)
    assert q == pytest.approx(-1.0)

    future_index, delta, reward, q = mac_success_targets(
        frame_index=4,
        horizon_idx=48,
        episode_length=5,
    )
    assert (future_index, delta) == (4, 0)
    assert reward == 0.0
    assert q == 0.0
