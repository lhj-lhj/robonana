from __future__ import annotations

from robonana.configs.posttrain_config import apply_iterative_posttrain_config


def test_posttrain_config_builds_four_separate_pool_views(monkeypatch, tmp_path):
    replay_root = tmp_path / "replay"
    checkpoint = tmp_path / "checkpoint" / "transformer" / "model.bin"
    monkeypatch.setenv("ROBONANA_REPLAY_ROOT", str(replay_root))
    monkeypatch.setenv("ROBONANA_POSTTRAIN_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("ROBONANA_COLLECTION_ROUND", "3")
    base = {
        "project_dir": str(tmp_path / "experiment"),
        "dataloaders": {
            "train": {
                "data_or_config": {
                    "_class_name": "RoboTwinHDF5Dataset",
                    "data_path": str(tmp_path / "original"),
                    "stats_path": str(tmp_path / "stats.json"),
                    "dino_online": True,
                },
                "sampler": {},
            }
        },
        "models": {"dino_dim": 3072},
        "train": {"loss_weights": {"dino_loss": 0.1}},
    }
    config = apply_iterative_posttrain_config(base)
    pools = config["dataloaders"]["train"]["data_or_config"]
    assert [pool["pool_name"] for pool in pools] == [
        "original_success",
        "collected_success_replay",
        "historical_failure_replay",
        "latest_failure",
    ]
    assert pools[2]["round_max"] == 2
    assert pools[3]["round_id"] == 3
    assert pools[1]["require_final_observation"] is True
    assert pools[2]["require_final_observation"] is True
    assert pools[3]["require_final_observation"] is True
    assert all(pool["dino_image_size"] == (480, 640) for pool in pools)
    assert config["dataloaders"]["train"]["sampler"]["pool_weights"] == {
        "original_success": 0.25,
        "collected_success_replay": 0.25,
        "historical_failure_replay": 0.25,
        "latest_failure": 0.25,
    }
    assert config["models"]["initialization"] == "trained"
    assert config["models"]["checkpoint"] == str(checkpoint)
    posttrain = config["train"]["posttrain"]
    assert posttrain["ema"]["decay"] == 0.995
    assert posttrain["failure_policy_improvement"]["candidate_count"] == 8
    assert posttrain["failure_policy_improvement"]["use_advantage_gate"] is False
    assert posttrain["td"]["next_action_policy"] == "ema"
    assert posttrain["td"]["bootstrap_failure_timeout"] is True
    assert config["train"]["q_target_mode"] == "td_posttrain"
    assert config["train"]["checkpoint_save_optimizer"] is True
