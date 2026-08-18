import sys
from types import SimpleNamespace

import pytest

from robonana.training.tracking import finish_wandb, init_wandb, log_wandb


class FakeRun:
    def __init__(self):
        self.logged = []
        self.exit_code = None

    def log(self, metrics, *, step):
        self.logged.append((metrics, step))

    def finish(self, *, exit_code):
        self.exit_code = exit_code


def test_disabled_tracking_does_not_import_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert init_wandb(project="p", entity="e", run_name=None, mode="disabled", config={}) is None


def test_online_tracking_initializes_logs_and_finishes(monkeypatch):
    run = FakeRun()
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    initialized = init_wandb(
        project="robonana",
        entity="hongjia-liu-aalto-university",
        run_name="unit-test",
        mode="online",
        config={"batch_size": 1},
    )
    log_wandb(initialized, {"train/loss": 1.25}, step=3)
    finish_wandb(initialized)

    assert calls == [
        {
            "project": "robonana",
            "entity": "hongjia-liu-aalto-university",
            "name": "unit-test",
            "mode": "online",
            "config": {"batch_size": 1},
        }
    ]
    assert run.logged == [({"train/loss": 1.25}, 3)]
    assert run.exit_code == 0


def test_tracking_rejects_unknown_mode():
    with pytest.raises(ValueError, match="wandb mode"):
        init_wandb(project="p", entity="e", run_name=None, mode="invalid", config={})
