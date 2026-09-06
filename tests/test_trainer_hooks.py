from types import SimpleNamespace

import pytest
import torch

from fact_train import Trainer
from robonana.training.posttraining import ValueExpertEMA
from robonana.training.robotwin_trainer import RoboNanaTrainer, resolve_cuda_device_index


def test_cuda_device_without_index_uses_current_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    assert resolve_cuda_device_index(torch.device("cuda")) == 7
    assert resolve_cuda_device_index(torch.device("cuda:3")) == 3
    assert resolve_cuda_device_index(torch.device("cpu")) is None


def test_pixel_eval_runs_only_after_backward_and_optimizer(monkeypatch):
    events = []
    trainer = object.__new__(RoboNanaTrainer)
    trainer.accelerator = SimpleNamespace(
        sync_gradients=True,
        is_main_process=False,
        unwrap_model=lambda model, **kwargs: model,
    )
    trainer._models = [torch.nn.Linear(1, 1)]
    trainer._optimizers = []
    trainer.target_value_ema = None
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


def _ema_hook_trainer(*, sync_gradients: bool, skipped: bool = False):
    trainer = object.__new__(RoboNanaTrainer)
    value_expert = torch.nn.Linear(2, 2, bias=False)
    online = SimpleNamespace(value_expert=value_expert)
    optimizer = SimpleNamespace(
        step_was_skipped=skipped,
        zero_grad=lambda: None,
    )
    trainer.accelerator = SimpleNamespace(
        sync_gradients=sync_gradients,
        is_main_process=False,
        unwrap_model=lambda model, **kwargs: model,
    )
    trainer._models = [online]
    trainer._optimizers = [optimizer]
    trainer.target_value_ema = ValueExpertEMA(value_expert, decay=0.995)
    trainer._cur_step = 1
    trainer._optimizer_step_succeeded = False
    return trainer


def test_gradient_accumulation_and_skipped_step_update_value_ema_only(monkeypatch):
    monkeypatch.setattr(Trainer, "backward_step", lambda self, loss: None)
    trainer = _ema_hook_trainer(sync_gradients=False)
    trainer.backward_step(torch.tensor(1.0))
    assert trainer.target_value_ema.update_count == 0

    trainer.accelerator.sync_gradients = True
    trainer.backward_step(torch.tensor(1.0))
    assert trainer.target_value_ema.update_count == 1

    skipped = _ema_hook_trainer(sync_gradients=True, skipped=True)
    skipped.backward_step(torch.tensor(1.0))
    assert skipped.target_value_ema.update_count == 0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("sync", [False, True])
def test_nonfinite_microstep_aborts_before_optimizer_and_ema(monkeypatch, bad_value, sync):
    events = []
    trainer = _ema_hook_trainer(sync_gradients=sync)
    trainer._optimizers[0].zero_grad = lambda: events.append("zero")
    monkeypatch.setattr(
        Trainer,
        "backward_step",
        lambda self, loss: events.append("optimizer"),
    )

    with pytest.raises(FloatingPointError, match="aborting all ranks"):
        trainer.backward_step(torch.tensor(bad_value))
    assert events == []
    assert trainer.target_value_ema.update_count == 0


def test_remote_bad_rank_uses_supported_sum_and_aborts_healthy_rank(monkeypatch):
    trainer = _ema_hook_trainer(sync_gradients=True)
    trainer.accelerator.num_processes = 2
    def reduce(flag, reduction):
        assert reduction == "sum"
        assert flag.item() == 0  # This rank is healthy; its peer is not.
        return flag + 1
    trainer.accelerator.reduce = reduce
    monkeypatch.setattr(Trainer, "backward_step", lambda *args: pytest.fail("must not step"))
    with pytest.raises(FloatingPointError):
        trainer.backward_step(torch.tensor(1.0))
    assert trainer.target_value_ema.update_count == 0


def test_value_target_ema_contains_no_flux_or_q_parameters():
    value = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 1))
    target = ValueExpertEMA(value, decay=0.9)
    assert set(target.state_dict()) == set(value.state_dict())
    assert all("flux" not in name and "q_expert" not in name for name in target.state_dict())


def test_fresh_critic_target_exact_copies_inherited_online_value():
    trainer = object.__new__(RoboNanaTrainer)
    online_value = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        online_value.weight.copy_(torch.tensor([[1.0, -2.0, 3.0]]))
    trainer._models = [SimpleNamespace(value_expert=online_value)]
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.mac_enabled = True
    trainer.mac_phase = "critic"
    trainer.with_ema = False
    trainer.posttrain_config = {
        "ema": {
            "decay": 0.995,
            "update_every_optimizer_steps": 1,
            "start_step": 0,
        }
    }

    trainer.set_ema_models()

    torch.testing.assert_close(
        trainer.target_value_ema.model.weight,
        online_value.weight.float(),
    )
    assert trainer.target_value_ema.update_count == 0


def test_critic_resume_rejects_missing_value_ema_files(monkeypatch, tmp_path):
    trainer = object.__new__(RoboNanaTrainer)
    trainer.target_value_ema = ValueExpertEMA(torch.nn.Linear(2, 1), decay=0.995)
    monkeypatch.setattr(Trainer, "load_model_hook", lambda self, models, input_dir: None)

    with pytest.raises(FileNotFoundError, match="critic resume checkpoint is incomplete"):
        trainer.load_model_hook([], str(tmp_path))


def test_posttrain_pixel_eval_resolves_the_sample_owning_pool():
    children = [
        SimpleNamespace(
            pool_name=name,
            eval_horizons=(12, 24, 48),
            load_eval_future_latents=lambda *args: None,
        )
        for name in (
            "original_success",
            "collected_success_replay",
            "historical_failure_replay",
            "latest_failure",
        )
    ]
    trainer = object.__new__(RoboNanaTrainer)
    trainer._dataloaders = [SimpleNamespace(dataset=SimpleNamespace(datasets=children))]

    assert trainer._pixel_eval_dataset(0) is children[0]
    assert trainer._pixel_eval_dataset(3) is children[3]
