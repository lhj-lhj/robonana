"""Warm-start direct reward/success heads from the 150k reward/Q run."""

from __future__ import annotations

import copy
import os
from pathlib import Path

from .robotwin_flux2_4b_dino import config as _base_config


config = copy.deepcopy(_base_config)
repo_root = Path(__file__).resolve().parents[3]

source_run = Path(
    os.environ.get(
        "ROBONANA_SOURCE_RUN",
        repo_root
        / "experiments"
        / "robotwin_flux2_4b_dino_reward_q_from120k_plus30k",
    )
)
source_checkpoint = Path(
    os.environ.get(
        "ROBONANA_TRAINED_CHECKPOINT",
        source_run
        / "models"
        / "checkpoint_epoch_7_step_150000"
        / "transformer"
        / "diffusion_pytorch_model.bin",
    )
)
source_config = Path(
    os.environ.get("ROBONANA_TRAINED_CONFIG", source_run / "config.json")
)

start_step = int(os.environ.get("ROBONANA_START_STEP", "150000"))
additional_steps = int(os.environ.get("ROBONANA_ADDITIONAL_STEPS", "10000"))
if start_step < 0:
    raise ValueError("ROBONANA_START_STEP cannot be negative")
if additional_steps <= 0:
    raise ValueError("ROBONANA_ADDITIONAL_STEPS must be positive")

config["project_dir"] = os.environ.get(
    "ROBONANA_PROJECT_DIR",
    str(
        repo_root
        / "experiments"
        / "robotwin_flux2_4b_dino_reward_success_q_from150k_plus10k"
    ),
)
if Path(config["project_dir"]).expanduser().resolve() == source_run.expanduser().resolve():
    raise ValueError("continuation project_dir must differ from the immutable 150k source run")

config["launch"]["gpu_ids"] = [
    int(value)
    for value in os.environ.get("ROBONANA_GPU_IDS", "6").split(",")
    if value.strip()
]
if not config["launch"]["gpu_ids"]:
    raise ValueError("ROBONANA_GPU_IDS must select at least one GPU")
config["dataloaders"]["train"].update(
    batch_size_per_gpu=int(os.environ.get("ROBONANA_BATCH_SIZE", "32"))
)
config["models"].update(
    initialization="trained",
    checkpoint=str(source_checkpoint),
    checkpoint_config=str(source_config),
    reward_head_type="direct",
    success_dim=1,
)
config["schedulers"].update(
    warmup_steps=int(os.environ.get("ROBONANA_WARMUP_STEPS", "500")),
    decay_steps=additional_steps,
)
config["train"].update(
    max_steps=start_step + additional_steps,
    initial_global_step=start_step,
    gradient_accumulation_steps=1,
    resume=False,
    checkpoint_interval=int(os.environ.get("ROBONANA_CHECKPOINT_INTERVAL", "1000")),
    checkpoint_total_limit=int(os.environ.get("ROBONANA_CHECKPOINT_TOTAL_LIMIT", "2")),
    early_checkpoint_steps=(start_step + 100,),
    pixel_eval_interval=int(os.environ.get("ROBONANA_PIXEL_EVAL_INTERVAL", "2000")),
)
config["train"]["tracker_init_kwargs"]["wandb"].update(
    name=os.environ.get(
        "WANDB_NAME", "flux4b-dino-reward-success-q-from150k-plus10k"
    )
)
