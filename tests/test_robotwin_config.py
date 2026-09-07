from __future__ import annotations

import importlib
import sys
import pytest


@pytest.mark.parametrize("key", ["ROBONANA_BATCH_SIZE", "ROBONANA_GRADIENT_ACCUMULATION_STEPS"])
def test_training_config_rejects_nonpositive_batch_settings(monkeypatch, key):
    monkeypatch.setenv(key, "0")
    module_name = "robonana.configs.robotwin_flux2"
    sys.modules.pop(module_name, None)
    try:
        with pytest.raises(ValueError, match="must be positive"):
            importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("phase", ["world_policy", "critic"])
@pytest.mark.parametrize("override", [False, True])
def test_mac_batch16_defaults_and_explicit_microbatch_override(monkeypatch, phase, override):
    from robonana.configs.posttrain_config import apply_mac_posttrain_config
    for key in ("ROBONANA_BATCH_SIZE", "ROBONANA_GRADIENT_ACCUMULATION_STEPS", "ROBONANA_GPU_IDS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ROBONANA_MAC_PHASE", phase)
    monkeypatch.delenv("ROBONANA_MIXED_PRECISION", raising=False)
    if override:
        monkeypatch.setenv("ROBONANA_BATCH_SIZE", "4")
        monkeypatch.setenv("ROBONANA_GRADIENT_ACCUMULATION_STEPS", "2")
    module_name = "robonana.configs.robotwin_flux2"
    sys.modules.pop(module_name, None)
    try:
        config = apply_mac_posttrain_config(importlib.import_module(module_name).config)
        batch = config["dataloaders"]["train"]["batch_size_per_gpu"]
        accumulation = config["train"]["gradient_accumulation_steps"]
        assert config["launch"]["gpu_ids"] == [6, 7]
        assert (batch, accumulation) == ((4, 2) if override else (8, 1))
        assert batch * len(config["launch"]["gpu_ids"]) * accumulation == 16
        assert config["train"]["mixed_precision"] == "no"
    finally:
        sys.modules.pop(module_name, None)


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
        assert module.config["models"]["reward_dim"] == 48
        assert module.config["models"]["success_dim"] == 1
        assert module.config["models"]["reward_head_type"] == "binary_chunk"
        assert module.config["models"]["q_dim"] == 1
        assert module.config["train"]["discount"] == 0.999
        assert module.config["train"]["reward_non_goal"] == -1.0
        assert module.config["train"]["reward_goal"] == 0.0
        assert module.config["train"]["q_target_mode"] == "mac_mot_v2"
    finally:
        sys.modules.pop(module_name, None)
