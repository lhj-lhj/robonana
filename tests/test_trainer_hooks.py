from types import SimpleNamespace

import torch

from fact_train import Trainer
from robonana.training.robotwin_trainer import RoboNanaTrainer, resolve_cuda_device_index


def test_cuda_device_without_index_uses_current_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    assert resolve_cuda_device_index(torch.device("cuda")) == 7
    assert resolve_cuda_device_index(torch.device("cuda:3")) == 3
    assert resolve_cuda_device_index(torch.device("cpu")) is None


def test_pixel_eval_runs_only_after_backward_and_optimizer(monkeypatch):
    events = []
    trainer = object.__new__(RoboNanaTrainer)
    trainer.accelerator = SimpleNamespace(sync_gradients=True)
    trainer._pending_pixel_eval = {"sample": torch.tensor(1)}
    trainer._optimizer_step_succeeded = False

    monkeypatch.setattr(Trainer, "backward_step", lambda self, loss: events.append("optimizer"))
    monkeypatch.setattr(Trainer, "print_step", lambda self: events.append("log"))
    monkeypatch.setattr(
        RoboNanaTrainer,
        "_run_fixed_horizon_eval",
        lambda self, payload: events.append("eval"),
    )

    trainer.backward_step(torch.tensor(1.0))
    assert events == ["optimizer"]
    trainer.print_step()
    assert events == ["optimizer", "eval", "log"]
    assert trainer._pending_pixel_eval is None


def test_early_checkpoint_reuses_fact_save_path(monkeypatch):
    observed_intervals = []
    trainer = object.__new__(RoboNanaTrainer)
    trainer.kwargs = {"early_checkpoint_steps": (200,)}
    trainer._cur_step = 200
    trainer.checkpoint_interval = 1000

    monkeypatch.setattr(
        Trainer,
        "save_checkpoint_step",
        lambda self: observed_intervals.append(self.checkpoint_interval),
    )

    trainer.save_checkpoint_step()
    assert observed_intervals == [1]
    assert trainer.checkpoint_interval == 1000
