"""Canonical pretrained FLUX.2 Klein 4B RoboNana with DINO supervision."""

from __future__ import annotations

import copy
import os
from pathlib import Path

from robonana.data import robotwin_lerobot as _robotwin_lerobot  # noqa: F401

from .robotwin_flux2 import config as _klein4b_config


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


config = copy.deepcopy(_klein4b_config)
repo_root = Path(__file__).resolve().parents[3]
dataset_root = Path(
    os.environ.get(
        "ROBONANA_DATASET_ROOT",
        "/workspace/datasets/fact-robotwin-v2/RoboTwin",
    )
)
max_steps = int(os.environ.get("ROBONANA_MAX_STEPS", "120000"))
num_workers = int(os.environ.get("ROBONANA_NUM_WORKERS", "4"))

config["project_dir"] = os.environ.get(
    "ROBONANA_PROJECT_DIR",
    str(repo_root / "experiments" / "robotwin_flux2_4b_dino_bs256_120k"),
)
config["launch"]["gpu_ids"] = [
    int(value)
    for value in os.environ.get("ROBONANA_GPU_IDS", "0,1,2,3,4,5,6,7").split(",")
    if value.strip()
]
config["dataloaders"]["train"].update(
    data_or_config=dict(
        _class_name="RoboTwinLeRobotDataset",
        data_path=str(dataset_root),
        stats_path=str(dataset_root / "robonana_norm_stats.json"),
        index_path=str(dataset_root / "robonana_index.json"),
        task_globs=("Clean/*", "Randomized/*"),
        action_chunk=48,
        action_dim=14,
        max_horizon=48,
        eval_horizons=(12, 24, 48),
        discount=float(os.environ.get("ROBONANA_DISCOUNT", "0.999")),
        reward_non_goal=float(os.environ.get("ROBONANA_REWARD_NON_GOAL", "-1.0")),
        reward_goal=float(os.environ.get("ROBONANA_REWARD_GOAL", "0.0")),
        q_target_mode=os.environ.get("ROBONANA_Q_TARGET_MODE", "mc_success"),
        dino_online=True,
    ),
    batch_size_per_gpu=int(os.environ.get("ROBONANA_BATCH_SIZE", "16")),
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    prefetch_factor=4 if num_workers > 0 else None,
    sampler=dict(type="RoboTwinEpisodeSampler", infinite=True),
)
config["models"].update(
    initialization="pretrained",
    checkpoint=_klein4b_config["models"]["checkpoint"],
    checkpoint_dir=_klein4b_config["models"]["checkpoint_dir"],
    params=copy.deepcopy(_klein4b_config["models"]["params"]),
    train_mode="full",
    gradient_checkpointing=False,
    dino_dim=3072,
    dino_encoder_model="vit_base_patch16_dinov3.lvd1689m",
    dino_encoder_batch_size=int(
        os.environ.get("ROBONANA_DINO_ENCODER_BATCH_SIZE", "48")
    ),
)
config["optimizers"].update(
    lr=float(os.environ.get("ROBONANA_BACKBONE_LR", "2e-5")),
    robot_lr=float(os.environ.get("ROBONANA_ROBOT_LR", "1e-4")),
)
config["schedulers"].update(
    warmup_steps=int(os.environ.get("ROBONANA_WARMUP_STEPS", "500")),
    decay_steps=max_steps,
)
config["train"].update(
    max_steps=max_steps,
    gradient_accumulation_steps=2,
    mixed_precision="bf16",
    activation_checkpointing=False,
    resume=_env_flag("ROBONANA_RESUME", True),
    pixel_eval_interval=int(os.environ.get("ROBONANA_PIXEL_EVAL_INTERVAL", "2000")),
    checkpoint_interval=int(os.environ.get("ROBONANA_CHECKPOINT_INTERVAL", "1000")),
    early_checkpoint_steps=tuple(
        int(value)
        for value in os.environ.get("ROBONANA_EARLY_CHECKPOINT_STEPS", "100").split(",")
        if value.strip()
    ),
)
config["train"]["loss_weights"]["dino_loss"] = float(
    os.environ.get("ROBONANA_DINO_LOSS_WEIGHT", "0.1")
)
