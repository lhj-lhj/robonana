from robonana.configs.posttrain_config import apply_mac_posttrain_config
import pytest
from robonana.normalization import A_STATS_PATH


def _base(tmp_path):
    return {
        "project_dir": str(tmp_path),
        "dataloaders": {"train": {"data_or_config": {
            "_class_name": "RoboTwinLeRobotDataset",
            "data_path": str(tmp_path / "data"),
            "stats_path": str(tmp_path / "stats.json"),
        }, "sampler": {}}},
        "models": {},
        "train": {"loss_weights": {}, "tracker_init_kwargs": {"wandb": {}}},
    }


def test_mac_rejects_hdf5_original_before_dataset_construction(tmp_path):
    base = _base(tmp_path)
    base["dataloaders"]["train"]["data_or_config"]["_class_name"] = "RoboTwinHDF5Dataset"
    with pytest.raises(ValueError, match="original data must use RoboTwinLeRobotDataset"):
        apply_mac_posttrain_config(base)


def test_mac_posttrain_defaults_to_fixed48_and_1000_step_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBONANA_REPLAY_ROOT", str(tmp_path / "replay"))
    config = apply_mac_posttrain_config(_base(tmp_path))
    assert config["models"]["architecture_version"] == "mac_mot_v2"
    assert config["models"]["initialization"] == "trained"
    assert config["models"]["reward_dim"] == 48
    assert "checkpoint_epoch_1_step_1000" in config["models"]["checkpoint"]
    assert config["train"]["q_target_mode"] == "mac_mot_v2"
    assert config["train"]["posttrain"]["chunk_horizon"] == 48
    assert config["train"]["posttrain"]["ema"]["target"] == "value_expert_only"
    assert "forward_autocast_dtype" not in config["train"]["posttrain"]["ema"]
    assert all(pool["stats_path"] == str(A_STATS_PATH)
               for pool in config["dataloaders"]["train"]["data_or_config"])


def test_replay_cannot_override_normalization_a(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBONANA_REPLAY_STATS_PATH", "/old/B.json")
    with pytest.raises(ValueError, match="Only Stage-1"):
        apply_mac_posttrain_config(_base(tmp_path))


def test_mac_critic_phase_only_changes_expert_training_surface(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBONANA_REPLAY_ROOT", str(tmp_path / "replay"))
    monkeypatch.setenv("ROBONANA_MAC_PHASE", "critic")
    config = apply_mac_posttrain_config(_base(tmp_path))
    assert config["models"]["train_mode"] == "critic"
    assert config["train"]["posttrain"]["phase"] == "critic"


@pytest.mark.parametrize("phase,expected", [("world_policy", 20000), ("critic", 10000)])
def test_phase_budget_overrides_base_and_aligns_decay(monkeypatch, tmp_path, phase, expected):
    for key in ("ROBONANA_MAX_STEPS", "ROBONANA_MAC_WORLD_POLICY_STEPS",
                "ROBONANA_MAC_TRAIN_STEPS", "ROBONANA_MAC_CRITIC_STEPS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ROBONANA_MAC_PHASE", phase)
    base = _base(tmp_path)
    base["train"]["max_steps"] = 150000
    base["schedulers"] = {"warmup_steps": 500, "decay_steps": 150000}
    config = apply_mac_posttrain_config(base)
    assert config["train"]["max_steps"] == expected
    assert config["schedulers"] == {"warmup_steps": 500, "decay_steps": expected}
    assert base["train"]["max_steps"] == 150000


def test_phase_budget_explicit_override_updates_scheduler(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBONANA_MAC_PHASE", "critic")
    monkeypatch.setenv("ROBONANA_MAC_CRITIC_STEPS", "12000")
    monkeypatch.delenv("ROBONANA_MAX_STEPS", raising=False)
    config = apply_mac_posttrain_config(_base(tmp_path))
    assert config["train"]["max_steps"] == config["schedulers"]["decay_steps"] == 12000
    monkeypatch.setenv("ROBONANA_MAX_STEPS", "8000")
    config = apply_mac_posttrain_config(_base(tmp_path))
    assert config["train"]["max_steps"] == config["schedulers"]["decay_steps"] == 8000
