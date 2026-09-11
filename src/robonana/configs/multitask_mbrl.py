"""中文：50任务固定协议；复用 MAC 配置，不改变单任务默认入口。
English: Fixed 50-task protocol composed from the maintained MAC configuration.
"""
import os
from pathlib import Path

from .posttrain_config import apply_mac_posttrain_config

MILESTONES = {"pretrain": (10000, 30000, 60000, 120000),
              "stage1": (10000, 30000, 60000), "stage2": (10000, 20000)}


def build_protocol_config(base, phase):
    if phase not in MILESTONES:
        raise ValueError("Protocol phase must be pretrain, stage1 or stage2")
    mac_phase = "critic" if phase == "stage2" else "world_policy"
    result = apply_mac_posttrain_config(base)
    result["train"]["posttrain"]["phase"] = mac_phase
    milestones = MILESTONES[phase]
    loader, train = result["dataloaders"]["train"], result["train"]
    pools = loader["data_or_config"]
    pools[0]["task_globs"] = ("Clean/*", "Randomized/*")
    # 中文：只从示范和本轮失败池取样；保留现有四池接口，零权重池不参与。
    # English: Reuse the four-pool sampler, explicitly disable unused replay pools.
    weights = dict(original_success=0.5, collected_success_replay=0.0,
                   historical_failure_replay=0.0, latest_failure=0.5)
    loader["sampler"]["pool_weights"] = weights
    train["posttrain"]["data_mixture"].update(weights)
    if phase == "pretrain":
        loader.update(data_or_config=pools[0],
                      sampler=dict(type="RoboTwinEpisodeSampler", infinite=True))
        train["posttrain"]["data_mixture"].update({key: float(key == "original_success") for key in weights})
        result["models"].update(initialization="flux_backbone",
                               checkpoint=base["models"]["checkpoint"], checkpoint_config=None)
    elif not os.environ.get("ROBONANA_MAC_PRETRAIN_CHECKPOINT") or not os.environ.get("ROBONANA_MAC_PRETRAIN_CONFIG"):
        raise ValueError("Stage1/Stage2 require explicit source checkpoint AND config; no historical fallback")
    if phase != "pretrain" and not os.environ.get("ROBONANA_REPLAY_ROOT"):
        raise ValueError("Stage1/Stage2 require explicit failure replay root")
    result["models"]["train_mode"] = mac_phase
    result["launch"]["gpu_ids"] = list(range(8))
    loader["batch_size_per_gpu"] = 16
    train.update(max_steps=milestones[-1], gradient_accumulation_steps=1,
                 resume=False, allow_uncertified_pretrain=False, mixed_precision="bf16",
                 checkpoint_interval=1000, early_checkpoint_steps=(), checkpoint_keeps=list(milestones),
                 checkpoint_total_limit=2, checkpoint_save_optimizer=True, disable_checkpointing=False)
    result["optimizers"].update(lr=1e-4 if phase == "stage2" else 2e-5,
                                robot_lr=2e-5 if phase == "stage1" else 1e-4)
    result["schedulers"].update(warmup_steps=500, decay_steps=milestones[-1])
    # A new experiment has its own optimizer/LR clock. No implicit continuation.
    result["project_dir"] = str(Path(os.environ.get("ROBONANA_PROTOCOL_ROOT", "experiments/multitask_mbrl")) / phase)
    train["tracker_init_kwargs"]["wandb"].update(
        entity="hongjia-liu-aalto-university", name=f"multitask-mbrl-loop0-{phase}")
    train["posttrain"]["current_collection_round"] = 0
    for pool in pools[1:]:
        if "round_id" in pool:
            pool["round_id"] = 0
        if "round_max" in pool:
            pool["round_max"] = -1
    smoke_steps = int(os.environ.get("ROBONANA_PROTOCOL_SMOKE_STEPS", "0"))
    if smoke_steps:
        if not 1 <= smoke_steps <= 10:
            raise ValueError("Smoke budget must be 1..10 updates")
        train.update(max_steps=smoke_steps, disable_checkpointing=True, log_interval=1,
                     checkpoint_keeps=[])
        result["schedulers"].update(decay_steps=smoke_steps, warmup_steps=1)
        train["tracker_init_kwargs"]["wandb"]["name"] += "-smoke"
    return result


def _entry():
    from .robotwin_flux2 import config as base
    return build_protocol_config(base, os.environ.get("ROBONANA_PROTOCOL_PHASE", "pretrain"))


config = _entry()
