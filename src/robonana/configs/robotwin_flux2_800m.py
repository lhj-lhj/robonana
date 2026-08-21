"""Scratch ~800M RoboNana on full Clean+Randomized FACT RoboTwin-v2."""

from __future__ import annotations

import copy
import os
from pathlib import Path

from robonana.data import robotwin_lerobot as _robotwin_lerobot  # noqa: F401

from .robotwin_flux2 import config as _base_config


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


config = copy.deepcopy(_base_config)
repo_root = Path(__file__).resolve().parents[3]
dataset_root = Path(
    os.environ.get("ROBONANA_DATASET_ROOT", "/workspace/datasets/fact-robotwin-v2/RoboTwin")
)
max_steps = int(os.environ.get("ROBONANA_MAX_STEPS", "120000"))
num_workers = int(os.environ.get("ROBONANA_NUM_WORKERS", "4"))

config["project_dir"] = os.environ.get(
    "ROBONANA_PROJECT_DIR",
    str(repo_root / "experiments" / "robotwin_flux2_800m_full_bs256_120k"),
)
config["launch"] = dict(
    gpu_ids=[
        int(value)
        for value in os.environ.get("ROBONANA_GPU_IDS", "0,1,2,3,4,5,6,7").split(",")
        if value.strip()
    ],
    distributed_type="MULTI_GPU",
    executable=config["launch"]["executable"],
    until_completion=False,
)
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
        rollout_horizon=int(os.environ.get("ROBONANA_ROLLOUT_HORIZON", "24")),
        rollout_horizon_prob=float(os.environ.get("ROBONANA_ROLLOUT_HORIZON_PROB", "0.5")),
        eval_horizons=(12, 24, 48),
    ),
    batch_size_per_gpu=int(os.environ.get("ROBONANA_BATCH_SIZE", "32")),
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    prefetch_factor=4 if num_workers > 0 else None,
    sampler=dict(type="RoboTwinEpisodeSampler", infinite=True),
)
config["models"].update(
    initialization="scratch",
    params=dict(
        in_channels=128,
        context_in_dim=7680,
        hidden_size=1536,
        num_heads=12,
        depth=4,
        depth_single_blocks=16,
        axes_dim=[32, 32, 32, 32],
        theta=2000,
        mlp_ratio=3.0,
        use_guidance_embed=False,
    ),
    train_mode="full",
    gradient_checkpointing=False,
)
config["models"].pop("checkpoint", None)
config["optimizers"].update(lr=float(os.environ.get("ROBONANA_LR", "1e-4")))
config["schedulers"].update(
    warmup_steps=int(os.environ.get("ROBONANA_WARMUP_STEPS", "500")),
    decay_steps=max_steps,
)
config["train"].update(
    max_steps=max_steps,
    mixed_precision="bf16",
    activation_checkpointing=False,
    resume=_env_flag("ROBONANA_RESUME", True),
    pixel_eval_interval=int(os.environ.get("ROBONANA_PIXEL_EVAL_INTERVAL", "1000")),
    checkpoint_interval=int(os.environ.get("ROBONANA_CHECKPOINT_INTERVAL", "1000")),
    early_checkpoint_steps=tuple(
        int(value)
        for value in os.environ.get("ROBONANA_EARLY_CHECKPOINT_STEPS", "100").split(",")
        if value.strip()
    ),
)
