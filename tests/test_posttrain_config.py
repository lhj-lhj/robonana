from robonana.configs.posttrain_config import apply_mac_posttrain_config


def _base(tmp_path):
    return {
        "project_dir": str(tmp_path),
        "dataloaders": {"train": {"data_or_config": {
            "_class_name": "RoboTwinHDF5Dataset",
            "data_path": str(tmp_path / "data"),
            "stats_path": str(tmp_path / "stats.json"),
        }, "sampler": {}}},
        "models": {},
        "train": {"loss_weights": {}, "tracker_init_kwargs": {"wandb": {}}},
    }


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


def test_mac_critic_phase_only_changes_expert_training_surface(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBONANA_REPLAY_ROOT", str(tmp_path / "replay"))
    monkeypatch.setenv("ROBONANA_MAC_PHASE", "critic")
    config = apply_mac_posttrain_config(_base(tmp_path))
    assert config["models"]["train_mode"] == "critic"
    assert config["train"]["posttrain"]["phase"] == "critic"
