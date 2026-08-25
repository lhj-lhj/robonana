from __future__ import annotations

import json

import pytest
from accelerate import init_empty_weights
from flux2.model import Flux2Params

from robonana.configs.robotwin_flux2_800m import config
from robonana.configs.robotwin_flux2 import _resolve_pred_action_bidirectional
from robonana.models.flux2_fact import Flux2FACTModel


def test_800m_full_dataset_config():
    data = config["dataloaders"]["train"]
    assert data["data_or_config"]["_class_name"] == "RoboTwinLeRobotDataset"
    assert tuple(data["data_or_config"]["task_globs"]) == ("Clean/*", "Randomized/*")
    assert data["batch_size_per_gpu"] == 32
    assert config["train"]["max_steps"] == 120000
    assert config["train"]["pixel_eval_interval"] == 1000
    assert config["launch"]["distributed_type"] == "MULTI_GPU"
    assert config["models"]["gradient_checkpointing"] is False
    assert config["models"]["value_dim"] == 1
    assert config["models"]["pred_action_bidirectional"] is True


def test_800m_parameter_count():
    params = Flux2Params(**config["models"]["params"])
    with init_empty_weights():
        model = Flux2FACTModel(params, action_dim=14, state_dim=14, max_horizon=48)
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == 791_333_376


def test_existing_legacy_experiment_preserves_causal_pred_action(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBONANA_PRED_ACTION_BIDIRECTIONAL", raising=False)
    (tmp_path / "config.json").write_text(
        json.dumps({"models": {"action_dim": 14}}),
        encoding="utf-8",
    )
    assert _resolve_pred_action_bidirectional(tmp_path) is False


def test_existing_hybrid_experiment_and_override_are_strict(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBONANA_PRED_ACTION_BIDIRECTIONAL", raising=False)
    (tmp_path / "config.json").write_text(
        json.dumps({"models": {"pred_action_bidirectional": True}}),
        encoding="utf-8",
    )
    assert _resolve_pred_action_bidirectional(tmp_path) is True

    monkeypatch.setenv("ROBONANA_PRED_ACTION_BIDIRECTIONAL", "false")
    assert _resolve_pred_action_bidirectional(tmp_path) is False
    monkeypatch.setenv("ROBONANA_PRED_ACTION_BIDIRECTIONAL", "maybe")
    with pytest.raises(ValueError, match="ROBONANA_PRED_ACTION_BIDIRECTIONAL"):
        _resolve_pred_action_bidirectional(tmp_path)
