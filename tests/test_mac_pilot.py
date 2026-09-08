import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from robonana.configs.robotwin_flux2_4b_mac_pilot import apply_pilot_config


def load_probe():
    path = Path(__file__).resolve().parents[1] / "scripts/diagnostics/probe_mac_world_fit.py"
    spec = importlib.util.spec_from_file_location("world_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pilot_budget_retains_all_checkpoints_and_uses_warmup():
    base = {"train": {"posttrain": {"phase": "world_policy"}}, "schedulers": {}, "optimizers": {}}
    world = apply_pilot_config(base)
    assert world["train"]["max_steps"] == 5000
    assert world["train"]["checkpoint_total_limit"] >= 6
    assert world["train"]["early_checkpoint_steps"] == (500,)
    assert world["schedulers"]["warmup_steps"] == 250
    assert "max_steps" not in base["train"]
    base["train"]["posttrain"]["phase"] = "critic"
    critic = apply_pilot_config(base)
    assert critic["train"]["max_steps"] == 500
    assert critic["optimizers"]["lr"] == 1e-5


def test_probe_selects_legal_starts_including_last_window():
    class Dataset:
        records = [None, None]
        episode_starts = [0, 17]
        episode_stops = [17, 34]
        def __len__(self):
            return 34
    assert load_probe().probe_indices(Dataset()) == [0, 8, 16, 17, 25, 33]


def test_probe_metrics_measure_prediction_and_persistence_separately():
    sample = SimpleNamespace(future=torch.ones(1, 2, 4), future_state=torch.ones(1, 1, 14),
                             reward_logits=torch.full((1, 48), -20.), success_logit=torch.tensor([[-20.]]))
    target = dict(future_latents=torch.ones(2, 4), current_latents=torch.zeros(2, 4),
                  future_state=torch.ones(14), state=torch.zeros(14), reward_chunk=torch.zeros(48),
                  reward_chunk_mask=torch.ones(48).bool(), success=torch.zeros(1))
    metrics = load_probe().world_metrics(sample, target)
    assert metrics["future_latent_mse"] == 0
    assert metrics["future_state_mse"] == 0
    assert metrics["persistence_latent_mse"] == 1
    assert metrics["persistence_state_mse"] == 1
    assert metrics["reward_accuracy"] == 1
    assert metrics["success_probability"] < 1e-6
