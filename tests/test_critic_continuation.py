from pathlib import Path
from types import SimpleNamespace
import runpy

import pytest

# This configuration/scheduler adapter has no torch dependency. Exercise it on
# the Windows maintainer host too, without importing training/__init__.py.
_module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/robonana/training/continuation.py"))
build_critic_continuation = _module["build_critic_continuation"]
rebase_loaded_scheduler = _module["rebase_loaded_scheduler"]
build_world_policy_resume = _module["build_world_policy_resume"]


def test_world_policy_resume_only_changes_execution_and_restore_paths(tmp_path):
    import copy
    source = dict(
        project_dir=str(tmp_path / "old"),
        models=dict(train_mode="world_policy", gradient_checkpointing=True),
        launch=dict(gpu_ids=list(range(8))),
        dataloaders=dict(train=dict(batch_size_per_gpu=32, data_or_config=[{"data_path": "original"}])),
        optimizers=dict(lr=2e-5, betas=["__tuple__", 0.9, 0.95]),
        schedulers=dict(warmup_steps=500, decay_steps=20000),
        train=dict(max_steps=20000, mixed_precision="no", gradient_accumulation_steps=1,
                   posttrain=dict(phase="world_policy"),
                   tracker_init_kwargs=dict(wandb=dict(id="old", resume="must"))),
    )
    original = copy.deepcopy(source)
    kwargs = dict(checkpoint=tmp_path / "old/models/checkpoint_epoch_1_step_100",
                  source_config=tmp_path / "old/config.json", project_dir=tmp_path / "new")
    config = build_world_policy_resume(source, **kwargs)
    assert source == original
    for key in ("launch", "dataloaders", "schedulers"):
        assert config[key] == source[key]
    assert config["optimizers"] == dict(lr=2e-5, betas=(0.9, 0.95))
    assert config["train"]["max_steps"] == 20000
    assert config["train"]["gradient_accumulation_steps"] == 1
    assert config["train"]["mixed_precision"] == "no"
    assert config["models"]["gradient_checkpointing"] is False
    assert config["train"]["resume"] is True
    assert config["train"]["resume_from"] == str(kwargs["checkpoint"])
    assert config["train"]["rebase_scheduler_on_resume"] is False
    assert config["train"]["allow_uncertified_pretrain"] is False
    assert "id" not in config["train"]["tracker_init_kwargs"]["wandb"]
    with pytest.raises(ValueError, match="separate"):
        build_world_policy_resume(source, **{**kwargs, "project_dir": tmp_path / "old"})
    source["models"]["train_mode"] = "critic"
    with pytest.raises(ValueError, match="world_policy"):
        build_world_policy_resume(source, **kwargs)
    source["models"]["train_mode"] = "world_policy"
    source["train"]["mixed_precision"] = "bf16"
    with pytest.raises(ValueError, match="FP32"):
        build_world_policy_resume(source, **kwargs)


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
    from robonana.normalization import A_STATS_PATH
    assert config["dataloaders"]["train"]["data_or_config"] == [
        {"data_path": "unchanged", "stats_path": str(A_STATS_PATH)}]
    assert source["dataloaders"]["train"]["data_or_config"] == [{"data_path": "unchanged"}]
    assert config["dataloaders"]["train"]["batch_size_per_gpu"] == 8
    assert config["train"]["gradient_accumulation_steps"] == 1
    assert config["train"]["mixed_precision"] == "no"
    assert "forward_autocast_dtype" not in config["train"]["posttrain"]["ema"]
    assert "id" not in config["train"]["tracker_init_kwargs"]["wandb"]
    assert config["schedulers"] == dict(warmup_steps=250, decay_steps=10000)
    # Removed dataset features must not leak back from a saved run's metadata.
    source["dataloaders"]["train"]["data_or_config"][0].update(
        dino_online=False, dino_image_size=None, eval_horizons=[12, 24, 48])
    larger = build_critic_continuation(source, checkpoint=Path("source/ckpt"),
                                      source_config=Path("source/config.json"),
                                      project_dir=Path("larger-run"), max_steps=17000,
                                      batch_size_per_gpu=16)
    assert larger["dataloaders"]["train"]["batch_size_per_gpu"] == 16
    assert larger["train"]["gradient_accumulation_steps"] == 1
    assert larger["optimizers"] == config["optimizers"]
    assert larger["dataloaders"]["train"]["data_or_config"] == config["dataloaders"]["train"]["data_or_config"]
    assert "eval_horizons" in source["dataloaders"]["train"]["data_or_config"][0]
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
