"""Saved-phase execution adapters; FACT still owns state restoration."""

import copy
import math
from robonana.normalization import A_STATS_PATH


def restore_config_tuples(value):
    """Decode FACT's JSON tuple marker without changing the saved file."""
    if isinstance(value, dict):
        return {key: restore_config_tuples(item) for key, item in value.items()}
    if isinstance(value, list):
        if value and value[0] == "__tuple__":
            return tuple(restore_config_tuples(item) for item in value[1:])
        return [restore_config_tuples(item) for item in value]
    return value


def build_world_policy_resume(source, *, checkpoint, source_config, project_dir,
                             gradient_checkpointing=False):
    """中文：原样续训 Stage 1，仅切换激活重计算；不重置优化器或学习率。

    English: Resume Stage 1 with an activation-recomputation override only.
    Reuse FACT restore and the existing model toggle. Preserve data, batch,
    precision, GPU topology and the original schedule/budget; never use the
    critic continuation's schedule extension or two-GPU defaults here.
    """
    config = restore_config_tuples(copy.deepcopy(source))
    if (
        config["models"]["train_mode"] != "world_policy"
        or config["train"]["posttrain"]["phase"] != "world_policy"
    ):
        raise ValueError("resume requires a world_policy checkpoint")
    if config["train"]["mixed_precision"] != "no":
        raise ValueError("resume requires FP32 execution")
    if (
        project_dir.resolve() == source_config.parent.resolve()
        or checkpoint.resolve().is_relative_to(project_dir.resolve())
    ):
        raise ValueError("use a separate project directory; preserve the source experiment")
    config["project_dir"] = str(project_dir)
    config["models"].update(
        checkpoint=str(checkpoint / "transformer/diffusion_pytorch_model.bin"),
        checkpoint_config=str(source_config),
        gradient_checkpointing=bool(gradient_checkpointing),
    )
    config["train"].update(
        resume=True, resume_from=str(checkpoint), rebase_scheduler_on_resume=False,
        allow_uncertified_pretrain=False, activation_checkpointing=False,
        checkpoint_save_optimizer=True,
    )
    tracker = config["train"]["tracker_init_kwargs"]["wandb"]
    tracker.pop("id", None)
    tracker.pop("resume", None)
    tracker["name"] = project_dir.name
    return config


def build_critic_continuation(source, *, checkpoint, source_config, project_dir, max_steps,
                              batch_size_per_gpu=8):
    """Reuse replay and optimization settings, with current FP32/batch defaults.

    This is continuation of the same critic phase, so restore online Q/V,
    Value EMA, Adam moments, RNG and progress. It is NOT a new critic phase.
    The old metadata-only autocast selector is discarded; no tensor migration
    or BF16 execution branch is introduced.
    """
    config = restore_config_tuples(copy.deepcopy(source))
    if not isinstance(batch_size_per_gpu, int) or batch_size_per_gpu < 1:
        raise ValueError("batch_size_per_gpu must be a positive integer")
    if config["models"]["train_mode"] != "critic":
        raise ValueError("continuation requires a critic checkpoint")
    if max_steps <= config["train"]["max_steps"]:
        raise ValueError("max_steps must extend the source training budget")
    config["project_dir"] = str(project_dir)
    config["runners"] = ["robonana.training.robotwin_trainer.RoboNanaTrainer"]
    config["launch"]["gpu_ids"] = [6, 7]
    config["launch"]["until_completion"] = False
    config["models"]["checkpoint"] = str(checkpoint / "transformer/diffusion_pytorch_model.bin")
    config["models"]["checkpoint_config"] = str(source_config)
    config["dataloaders"]["train"]["batch_size_per_gpu"] = batch_size_per_gpu
    pools = config["dataloaders"]["train"]["data_or_config"]
    for pool in pools if isinstance(pools, (list, tuple)) else [pools]:
        # User-approved correction, not exact reproduction of the old run.
        # Only this new config is changed; the saved source config is untouched.
        pool["stats_path"] = str(A_STATS_PATH)
        # Normalize saved configuration metadata into the maintained schema;
        # removed DINO/pixel-eval implementations are not runtime alternatives.
        for removed in ("dino_online", "dino_image_size", "eval_horizons"):
            pool.pop(removed, None)
    config["train"].pop("pixel_eval_interval", None)
    config["train"].get("loss_weights", {}).pop("dino_loss", None)
    config["train"].update(
        max_steps=max_steps, gradient_accumulation_steps=1, mixed_precision="no",
        resume=True, resume_from=str(checkpoint), rebase_scheduler_on_resume=True,
        checkpoint_interval=1000, early_checkpoint_steps=(), checkpoint_total_limit=3,
        checkpoint_save_optimizer=True, log_interval=10, log_with="wandb",
    )
    config["train"]["posttrain"]["ema"].pop("forward_autocast_dtype", None)
    config["schedulers"]["decay_steps"] = max_steps
    tracker = config["train"]["tracker_init_kwargs"]["wandb"]
    tracker.pop("id", None)
    tracker.pop("resume", None)
    tracker["name"] = project_dir.name
    return config


def rebase_loaded_scheduler(wrapped, step):
    """Apply the extended FACT LambdaLR curve before the next Adam update.

    A completed source schedule restores LR=0. Keep last_epoch and optimizer
    moments, but recompute the LR with the new decay endpoint. Calling step()
    here would incorrectly advance the scheduler by one optimizer step.
    Reference: third_party/FACT/fact_train/trainers/trainer.py (resume) and
    https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
    """
    scheduler = getattr(wrapped, "scheduler", wrapped)
    if scheduler.last_epoch != step:
        raise ValueError(f"scheduler/global step mismatch: {scheduler.last_epoch} != {step}")
    rates = [base * fn(step) for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas, strict=True)]
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError(f"invalid continuation learning rates: {rates}")
    for group, rate in zip(scheduler.optimizer.param_groups, rates, strict=True):
        group["lr"] = rate
    scheduler._last_lr = rates
    return rates
