from __future__ import annotations

import pytest

from robonana.configs.posttrain_config import apply_mac_posttrain_config


def test_mac_posttrain_is_fixed_h1_and_uses_120k_migration(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBONANA_REPLAY_ROOT", str(tmp_path / "replay"))
    monkeypatch.setenv("ROBONANA_SOURCE_RUN", str(tmp_path / "run120k"))
    base = {
        "project_dir": str(tmp_path / "base"),
        "dataloaders": {
            "train": {
                "data_or_config": {
                    "_class_name": "RoboTwinLeRobotDataset",
                    "data_path": str(tmp_path / "original"),
                    "stats_path": str(tmp_path / "stats.json"),
                    "dino_online": True,
                },
                "sampler": {},
            }
        },
        "models": {},
        "train": {
            "loss_weights": {},
            "tracker_init_kwargs": {"wandb": {}},
        },
    }
    config = apply_mac_posttrain_config(base)
    assert config["models"]["architecture_version"] == "mac_mot_v2"
    assert config["models"]["train_mode"] == "world_policy"
    assert config["models"]["initialization"] == "mac_from_legacy"
    assert config["train"]["resume"] is True
    assert config["models"]["reward_dim"] == 48
    assert config["models"]["checkpoint"].endswith(
        "checkpoint_epoch_6_step_120000/transformer/diffusion_pytorch_model.bin"
    )
    posttrain = config["train"]["posttrain"]
    assert posttrain["imagination"]["rollout_chunks"] == 1
    assert posttrain["imagination"]["candidate_count"] == 8
    assert posttrain["imagination"]["fresh_each_batch"] is True
    assert posttrain["environment_policy"]["candidate_count"] == 32
    assert posttrain["ema"]["decay"] == 0.995
    assert posttrain["ema"]["target"] == "value_expert_only"
    assert posttrain["ema"]["initial_checkpoint"] == ""
    pools = config["dataloaders"]["train"]["data_or_config"]
    assert all(pool["fixed_horizon"] == 48 for pool in pools)
    assert all(pool["q_target_mode"] == "mac_mot_v2" for pool in pools)


def test_mac_critic_phase_freezes_flux_surface_in_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBONANA_REPLAY_ROOT", str(tmp_path / "replay"))
    monkeypatch.setenv("ROBONANA_SOURCE_RUN", str(tmp_path / "run120k"))
    monkeypatch.setenv("ROBONANA_MAC_PHASE", "critic")
    monkeypatch.setenv(
        "ROBONANA_MAC_TARGET_VALUE_CHECKPOINT", str(tmp_path / "target.safetensors")
    )
    monkeypatch.setenv(
        "ROBONANA_MAC_TARGET_VALUE_STATE", str(tmp_path / "target-state.json")
    )
    base = {
        "project_dir": str(tmp_path / "base"),
        "dataloaders": {"train": {"data_or_config": {
            "data_path": str(tmp_path / "original"),
            "stats_path": str(tmp_path / "stats.json"),
        }, "sampler": {}}},
        "models": {},
        "optimizers": {"lr": 1e-5, "robot_lr": 1e-4},
        "train": {"loss_weights": {}, "tracker_init_kwargs": {"wandb": {}}},
    }
    config = apply_mac_posttrain_config(base)
    assert config["models"]["train_mode"] == "critic"
    assert config["train"]["posttrain"]["phase"] == "critic"
    assert config["optimizers"]["lr"] == config["optimizers"]["robot_lr"]
    assert config["train"]["posttrain"]["ema"]["initial_checkpoint"] == str(
        tmp_path / "target.safetensors"
    )


def test_mac_posttrain_can_continue_from_an_exact_mac_checkpoint(
    monkeypatch, tmp_path
):
    checkpoint = (
        tmp_path
        / "round0"
        / "models"
        / "checkpoint"
        / "transformer"
        / "model.bin"
    )
    model_config = tmp_path / "round0" / "config.json"
    monkeypatch.setenv("ROBONANA_REPLAY_ROOT", str(tmp_path / "replay"))
    monkeypatch.setenv("ROBONANA_MAC_INITIALIZATION", "trained")
    monkeypatch.setenv("ROBONANA_MAC_PRETRAIN_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("ROBONANA_MAC_PRETRAIN_CONFIG", str(model_config))
    base = {
        "project_dir": str(tmp_path / "base"),
        "dataloaders": {
            "train": {
                "data_or_config": {
                    "_class_name": "RoboTwinLeRobotDataset",
                    "data_path": str(tmp_path / "original"),
                    "stats_path": str(tmp_path / "stats.json"),
                },
                "sampler": {},
            }
        },
        "models": {},
        "train": {
            "loss_weights": {},
            "tracker_init_kwargs": {"wandb": {}},
        },
    }

    config = apply_mac_posttrain_config(base)

    assert config["models"]["initialization"] == "trained"
    assert config["models"]["checkpoint"] == str(checkpoint)
    assert config["models"]["checkpoint_config"] == str(model_config)


def test_mac_trained_continuation_requires_explicit_lineage(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBONANA_REPLAY_ROOT", str(tmp_path / "replay"))
    monkeypatch.setenv("ROBONANA_MAC_INITIALIZATION", "trained")
    monkeypatch.delenv("ROBONANA_MAC_PRETRAIN_CHECKPOINT", raising=False)
    monkeypatch.delenv("ROBONANA_MAC_PRETRAIN_CONFIG", raising=False)
    base = {
        "project_dir": str(tmp_path / "base"),
        "dataloaders": {
            "train": {
                "data_or_config": {
                    "_class_name": "RoboTwinLeRobotDataset",
                    "data_path": str(tmp_path / "original"),
                    "stats_path": str(tmp_path / "stats.json"),
                },
                "sampler": {},
            }
        },
        "models": {},
        "train": {
            "loss_weights": {},
            "tracker_init_kwargs": {"wandb": {}},
        },
    }

    with pytest.raises(ValueError, match="requires explicit"):
        apply_mac_posttrain_config(base)
