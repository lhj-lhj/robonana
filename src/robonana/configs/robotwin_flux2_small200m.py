"""Scratch ~200M FLUX.2 RoboNana on the full RoboTwin training set."""

from __future__ import annotations

import copy
import os

from .robotwin_flux2 import INITIAL_DATA_CONFIG, config as _base_config


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


config = copy.deepcopy(_base_config)
max_steps = int(os.environ.get("ROBONANA_MAX_STEPS", "10000"))
num_workers = int(os.environ.get("ROBONANA_NUM_WORKERS", "4"))

config["project_dir"] = os.environ.get(
    "ROBONANA_PROJECT_DIR",
    str(config["project_dir"]).replace("robotwin_flux2", "robotwin_flux2_small200m"),
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

# Keep the original 50-task/2500-episode dataset, episode-uniform sampler,
# action-chunk construction, tail clipping, and mixed horizon distribution.
dataset_config = copy.deepcopy(INITIAL_DATA_CONFIG)
config["dataloaders"]["train"].update(
    data_or_config=dataset_config,
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
        hidden_size=1024,
        num_heads=8,
        depth=2,
        depth_single_blocks=8,
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
)
