"""Continue the legacy 120k 4B+DINO run with reward/Q for 30k steps."""

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
        / "robotwin_flux2_4b_dino_grouped_lr_A_bidir_G_causal_bs256_120k",
    )
)
source_checkpoint = Path(
    os.environ.get(
        "ROBONANA_TRAINED_CHECKPOINT",
        source_run
        / "models"
        / "checkpoint_epoch_6_step_120000"
        / "transformer"
        / "diffusion_pytorch_model.bin",
    )
)
source_config = Path(
    os.environ.get("ROBONANA_TRAINED_CONFIG", source_run / "config.json")
)

start_step = int(os.environ.get("ROBONANA_START_STEP", "120000"))
additional_steps = int(os.environ.get("ROBONANA_ADDITIONAL_STEPS", "30000"))
if start_step < 0:
    raise ValueError("ROBONANA_START_STEP cannot be negative")
if additional_steps <= 0:
    raise ValueError("ROBONANA_ADDITIONAL_STEPS must be positive")
max_steps = start_step + additional_steps

config["project_dir"] = os.environ.get(
    "ROBONANA_PROJECT_DIR",
    str(
        repo_root
        / "experiments"
        / "robotwin_flux2_4b_dino_reward_q_from120k_plus30k"
    ),
)
if Path(config["project_dir"]).expanduser().resolve() == source_run.expanduser().resolve():
    raise ValueError(
        "continuation project_dir must differ from the immutable 120k source run"
    )
config["models"].update(
    initialization="trained",
    checkpoint=str(source_checkpoint),
    checkpoint_config=str(source_config),
)
config["schedulers"].update(
    warmup_steps=int(os.environ.get("ROBONANA_WARMUP_STEPS", "500")),
    decay_steps=additional_steps,
)
config["train"].update(
    max_steps=max_steps,
    initial_global_step=start_step,
    resume=False,
    checkpoint_interval=int(os.environ.get("ROBONANA_CHECKPOINT_INTERVAL", "1000")),
    checkpoint_total_limit=int(os.environ.get("ROBONANA_CHECKPOINT_TOTAL_LIMIT", "2")),
    early_checkpoint_steps=(start_step + 100,),
    pixel_eval_interval=int(os.environ.get("ROBONANA_PIXEL_EVAL_INTERVAL", "2000")),
)
config["train"]["tracker_init_kwargs"]["wandb"].update(
    name=os.environ.get(
        "WANDB_NAME", "flux4b-dino-reward-q-from120k-plus30k"
    )
)
