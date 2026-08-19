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
    finally:
        sys.modules.pop(module_name, None)
