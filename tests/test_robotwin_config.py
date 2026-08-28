from __future__ import annotations

import importlib
import sys


def test_training_config_logically_mixes_separate_rollout_root(monkeypatch, tmp_path) -> None:
    initial_root = tmp_path / "initial"
    rollout_root = tmp_path / "rollouts" / "collection"
    monkeypatch.setenv("ROBONANA_DATASET_ROOT", str(initial_root))
    monkeypatch.setenv("ROBONANA_ROLLOUT_DATASET_ROOT", str(rollout_root))
    monkeypatch.setenv("ROBONANA_ROLLOUT_DATASET_WEIGHT", "2.5")
    module_name = "robonana.configs.robotwin_flux2"
    sys.modules.pop(module_name, None)
    try:
        module = importlib.import_module(module_name)
        train = module.config["dataloaders"]["train"]
        assert [row["data_path"] for row in train["data_or_config"]] == [
            str(initial_root),
            str(rollout_root),
        ]
        assert train["sampler"] == {
            "type": "RoboTwinMixtureSampler",
            "infinite": True,
            "dataset_weights": [1.0, 2.5],
        }
        assert module.config["models"]["params"]["hidden_size"] == 3072
        assert module.config["models"]["reward_dim"] == 1
        assert module.config["models"]["q_dim"] == 1
        assert module.config["train"]["discount"] == 0.999
        assert module.config["train"]["reward_non_goal"] == -1.0
        assert module.config["train"]["reward_goal"] == 0.0
        assert module.config["train"]["q_target_mode"] == "mc_success"
        assert module.config["train"]["loss_weights"]["reward_loss"] == 0.01
        assert module.config["train"]["loss_weights"]["q_loss"] == 0.001
    finally:
        sys.modules.pop(module_name, None)
