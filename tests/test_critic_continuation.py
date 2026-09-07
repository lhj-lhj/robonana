from pathlib import Path
from types import SimpleNamespace
import runpy

import pytest

# This configuration/scheduler adapter has no torch dependency. Exercise it on
# the Windows maintainer host too, without importing training/__init__.py.
_module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/robonana/training/continuation.py"))
build_critic_continuation = _module["build_critic_continuation"]
rebase_loaded_scheduler = _module["rebase_loaded_scheduler"]


def test_continuation_preserves_data_and_optimizer_with_current_execution():
    source = dict(models=dict(train_mode="critic"), launch={}, runners=["old"],
                  dataloaders=dict(train=dict(data_or_config=[{"data_path": "unchanged"}])),
                  optimizers=dict(betas=["__tuple__", 0.9, 0.95]),
                  schedulers=dict(warmup_steps=250, decay_steps=5000),
                  train=dict(max_steps=5000, posttrain=dict(ema=dict(forward_autocast_dtype="bfloat16")),
                             tracker_init_kwargs=dict(wandb=dict(id="old-id"))))
    config = build_critic_continuation(source, checkpoint=Path("source/ckpt"),
                                      source_config=Path("source/config.json"),
                                      project_dir=Path("new-run"), max_steps=10000)
    assert source["train"]["max_steps"] == 5000
    assert config["optimizers"]["betas"] == (0.9, 0.95)
    assert config["dataloaders"]["train"]["data_or_config"] == [{"data_path": "unchanged"}]
    assert config["dataloaders"]["train"]["batch_size_per_gpu"] == 8
    assert config["train"]["gradient_accumulation_steps"] == 1
    assert config["train"]["mixed_precision"] == "no"
    assert "forward_autocast_dtype" not in config["train"]["posttrain"]["ema"]
    assert "id" not in config["train"]["tracker_init_kwargs"]["wandb"]
    assert config["schedulers"] == dict(warmup_steps=250, decay_steps=10000)
    larger = build_critic_continuation(source, checkpoint=Path("source/ckpt"),
                                      source_config=Path("source/config.json"),
                                      project_dir=Path("larger-run"), max_steps=17000,
                                      batch_size_per_gpu=16)
    assert larger["dataloaders"]["train"]["batch_size_per_gpu"] == 16
    assert larger["train"]["gradient_accumulation_steps"] == 1
    assert larger["optimizers"] == config["optimizers"]
    assert larger["schedulers"]["decay_steps"] == 17000
    with pytest.raises(ValueError, match="positive integer"):
        build_critic_continuation(source, checkpoint=Path("source/ckpt"),
                                  source_config=Path("source/config.json"),
                                  project_dir=Path("bad-run"), max_steps=17000,
                                  batch_size_per_gpu=0)


def test_scheduler_rebase_does_not_advance_progress_or_replace_optimizer():
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}], state={"moment": object()})
    scheduler = SimpleNamespace(last_epoch=5000, base_lrs=[1e-5],
                                lr_lambdas=[lambda step: 0.5], optimizer=optimizer)
    state = optimizer.state
    assert rebase_loaded_scheduler(SimpleNamespace(scheduler=scheduler), 5000) == [5e-6]
    assert optimizer.param_groups[0]["lr"] == 5e-6
    assert optimizer.state is state
    assert scheduler.last_epoch == 5000
    with pytest.raises(ValueError, match="mismatch"):
        rebase_loaded_scheduler(scheduler, 5001)
    scheduler.lr_lambdas = [lambda step: 0.0]
    with pytest.raises(ValueError, match="invalid"):
        rebase_loaded_scheduler(scheduler, 5000)
