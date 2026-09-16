import h5py
import numpy as np
import pytest
import torch

from robonana.data.robotwin_hdf5 import (
    ALOHA_DELTA_MASK, EpisodeRecord, RoboTwinHDF5Dataset, mac_binary_chunk_targets,
)
from robonana.training.losses import masked_action_mse


def test_fixed_mac_reward_chunk_uses_absorbing_success_suffix():
    target, mask = mac_binary_chunk_targets(
        delta_steps=3, success_terminal=True, chunk_horizon=48
    )
    assert target.shape == (48,)
    assert mask.shape == (48,)
    torch.testing.assert_close(target[:3], torch.zeros(3))
    torch.testing.assert_close(target[3:], torch.ones(45))
    torch.testing.assert_close(mask, torch.ones(48))


def test_failed_tail_is_unknown_and_not_padded():
    target, mask = mac_binary_chunk_targets(
        delta_steps=48, success_terminal=False, chunk_horizon=48
    )
    torch.testing.assert_close(target, torch.zeros(48))
    torch.testing.assert_close(mask, torch.ones(48))


def episode_dataset(tmp_path, monkeypatch, *, source, success=True, length=11, mode="fixed48"):
    """Real adapter I/O; only frozen image/language caches are synthetic."""
    states = np.arange(length * 14, dtype=np.float32).reshape(length, 14) / 100
    states[:, 6], states[:, 13] = 1., .25
    actions = states + .2
    if source == "lerobot":
        import pandas as pd
        from robonana.data.robotwin_lerobot import RoboTwinLeRobotDataset
        actions[-1] = states[-1]  # Released FACT demonstrations' terminal row.
        path = tmp_path / "episode_000000.parquet"
        pd.DataFrame({"observation.state": list(states), "action": list(actions),
                      "frame_index": np.arange(length)}).to_parquet(path)
        cls = RoboTwinLeRobotDataset
    else:
        path = tmp_path / "episode0.hdf5"
        actions[-1] = actions[-2] if length > 1 else states[-1] + 5
        with h5py.File(path, "w") as handle:
            handle["joint_action/vector"] = states
            handle["policy_action/vector"] = actions
            handle["transition_valid"] = np.arange(length) < length - 1
        cls = RoboTwinHDF5Dataset
    ds = cls(str(tmp_path), stats_path="/unused", world_conditioning=mode, allow_empty=True)
    ds._set_records([EpisodeRecord("task", tmp_path, path, 0, length, success=success,
                                  has_final_observation=True, time_limit_truncated=not success)])
    mean = np.linspace(.1, 1.4, 14, dtype=np.float32)
    std = np.linspace(.5, 2., 14, dtype=np.float32)
    ds._stats = {"norm_stats": {key: dict(mean=mean, std=std) for key in ("action", "observation.state")}}
    monkeypatch.setattr(ds, "_latents", lambda _: torch.arange(length).float()[:, None, None].expand(-1, 2, 8))
    monkeypatch.setattr(ds, "_context", lambda _: torch.zeros(3, 16))
    return ds, states, actions, torch.from_numpy(mean), torch.from_numpy(std)


@pytest.mark.parametrize("mode", ["fixed48", "rope_prefix"])
@pytest.mark.parametrize("source", ["lerobot", "hdf5"])
def test_terminal_row_trains_zero_delta_hold_with_absolute_grippers(tmp_path, monkeypatch, source, mode):
    ds, states, _actions, mean, std = episode_dataset(tmp_path, monkeypatch, source=source, mode=mode)
    try:
        assert len(ds) == len(states)  # The terminal observation must be sampled.
        row = ds._get_data(len(ds) - 1)
        joint = torch.from_numpy(ALOHA_DELTA_MASK)
        raw_delta = row["action"] * std + mean
        torch.testing.assert_close(raw_delta[:, joint], torch.zeros(48, 12), atol=1e-6, rtol=0)
        torch.testing.assert_close(row["action"][:, joint], (-mean[joint] / std[joint]).expand(48, -1))
        torch.testing.assert_close(raw_delta[:, ~joint], torch.from_numpy(states[-1, ~ALOHA_DELTA_MASK]).expand(48, -1))
        assert row["action_valid_mask"].all() and row["action_loss_mask"].item() == 1
        assert row["delta_steps"].item() == 0 and row["success"].item() == 1
        assert torch.equal(row["current_latents"], row["future_latents"])
        assert row["reward_chunk"].eq(1).all()
        prediction = (row["action"] + 1).unsqueeze(0).detach().requires_grad_()
        loss = masked_action_mse(prediction, row["action"][None], row["action_valid_mask"][None], row["action_loss_mask"][None])
        loss.backward()
        assert torch.isfinite(prediction.grad).all() and torch.all(prediction.grad != 0)
    finally:
        ds.close()


@pytest.mark.parametrize("source", ["lerobot", "hdf5"])
def test_preterminal_padding_holds_terminal_pose_relative_to_current_state(tmp_path, monkeypatch, source):
    from fact_datasets.datasets.lerobot_dataset import LeRobotDataset
    ds, states, actions, mean, std = episode_dataset(tmp_path, monkeypatch, source=source)
    try:
        frame = len(states) - 3
        row = ds._get_data(frame)
        fact = object.__new__(LeRobotDataset)
        fact.delta_info = {"action": 48}
        indices = fact._get_query_indices(frame, len(states), "action")
        expected = actions[indices].copy()
        expected[indices == len(states) - 1] = states[-1]  # HDF5's terminal command is only a placeholder.
        expected[:, ALOHA_DELTA_MASK] -= states[frame, ALOHA_DELTA_MASK]
        torch.testing.assert_close(row["action"], (torch.from_numpy(expected) - mean) / std)
        # Before the goal, a constant terminal absolute pose is not zero delta.
        delta = row["action"] * std + mean
        assert delta[2:, ALOHA_DELTA_MASK].abs().sum() > 0
        absolute = delta.clone()
        absolute[:, ALOHA_DELTA_MASK] += torch.from_numpy(states[frame, ALOHA_DELTA_MASK])
        torch.testing.assert_close(absolute[2:], torch.from_numpy(states[-1]).expand(46, -1))
        assert row["action_valid_mask"].all()
        # Fancy indexing / padding must never mutate cached episode arrays.
        _, after = ds._episode_state_action(ds.records[0])
        np.testing.assert_array_equal(after, actions)
    finally:
        ds.close()


def test_failure_windows_and_bc_exclusion_are_unchanged(tmp_path, monkeypatch):
    ds, states, actions, mean, std = episode_dataset(tmp_path, monkeypatch, source="hdf5", success=False, length=61)
    try:
        assert len(ds) == 13
        row = ds._get_data(12)
        expected = actions[12:60].copy()
        expected[:, ALOHA_DELTA_MASK] -= states[12, ALOHA_DELTA_MASK]
        torch.testing.assert_close(row["action"], (torch.from_numpy(expected) - mean) / std)
        assert row["action_valid_mask"].all() and row["delta_steps"].item() == 48
        assert row["action_loss_mask"].item() == 0 and row["success"].item() == 0
        with pytest.raises(IndexError):
            ds._get_data(13)
        prediction = (row["action"] + 1).unsqueeze(0).detach().requires_grad_()
        loss = masked_action_mse(prediction, row["action"][None], row["action_valid_mask"][None], row["action_loss_mask"][None])
        loss.backward()
        assert loss.item() == 0 and torch.count_nonzero(prediction.grad) == 0
    finally:
        ds.close()


def test_short_failure_is_still_excluded(tmp_path, monkeypatch):
    ds, *_ = episode_dataset(tmp_path, monkeypatch, source="hdf5", success=False, length=48)
    assert ds.records == [] and ds.episode_stops.size == 0
    ds.close()


def test_success_mask_does_not_hide_missing_real_transitions(tmp_path, monkeypatch):
    ds, *_ = episode_dataset(tmp_path, monkeypatch, source="hdf5")
    valid = np.array([True] * 10 + [False])
    valid[5] = False
    monkeypatch.setattr(ds, "_episode_transition_valid", lambda _: valid)
    with pytest.raises(RuntimeError, match="missing transitions"):
        ds._get_data(0)
    ds.close()
