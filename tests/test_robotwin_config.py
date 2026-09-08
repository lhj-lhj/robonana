from __future__ import annotations

import importlib
import sys
import os
import json
import pytest


@pytest.mark.parametrize("phase", ["world_policy", "critic"])
def test_default_mac_entrypoint_constructs_real_datasets_clean_only(monkeypatch, tmp_path, phase):
    from robonana.data.robotwin_hdf5 import RoboTwinHDF5Dataset
    from robonana.data.robotwin_lerobot import RoboTwinLeRobotDataset

    for key in list(os.environ):
        if key.startswith("ROBONANA_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("ROBONANA_MAC_PHASE", phase)
    modules = ("robonana.configs.robotwin_flux2_4b_mac", "robonana.configs.robotwin_flux2")
    for name in modules:
        sys.modules.pop(name, None)
    try:
        config = importlib.import_module(modules[0]).config
        pools = config["dataloaders"]["train"]["data_or_config"]
        assert pools[0]["_class_name"] == "RoboTwinLeRobotDataset"
        assert pools[0]["data_path"].replace("\\", "/") == "/workspace/datasets/fact-robotwin-v2/RoboTwin"
        assert pools[0]["task_globs"] == ("Clean/hanging_mug",)
        assert "task_glob" not in pools[0]
        original = RoboTwinLeRobotDataset.load(pools[0])
        original.close()
        for pool in pools[1:]:
            assert pool["_class_name"] == "RoboTwinHDF5Dataset"
            assert "task_globs" not in pool
            RoboTwinHDF5Dataset.load(pool).close()

        # Exercise actual discovery with both source categories present. Empty
        # parquet placeholders suffice: this test indexes, never decodes frames.
        for category in ("Clean", "Randomized"):
            task = tmp_path / category / "hanging_mug"
            (task / "meta").mkdir(parents=True)
            (task / "meta/episodes.jsonl").write_text(
                json.dumps({"episode_index": 0, "length": 100}) + "\n", encoding="utf-8")
            parquet = task / "data/chunk-000/episode_000000.parquet"
            parquet.parent.mkdir(parents=True)
            parquet.touch()
        fixture = dict(pools[0], data_path=str(tmp_path), index_path=str(tmp_path / "index.json"))
        dataset = RoboTwinLeRobotDataset.load(fixture)
        try:
            dataset._ensure_index()
            assert len(dataset.records) == 1
            assert dataset.records[0].task_dir == (tmp_path / "Clean/hanging_mug").resolve()
        finally:
            dataset.close()
    finally:
        for name in modules:
            sys.modules.pop(name, None)


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
