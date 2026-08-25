"""Pretrained FLUX.2 Klein 4B RoboNana with an online DINOv3 target."""

from __future__ import annotations

import copy
import os
from pathlib import Path

from .robotwin_flux2 import config as _klein4b_config
from .robotwin_flux2_800m_dino import config as _dino_config


config = copy.deepcopy(_dino_config)
repo_root = Path(__file__).resolve().parents[3]

config["project_dir"] = os.environ.get(
    "ROBONANA_PROJECT_DIR",
    str(repo_root / "experiments" / "robotwin_flux2_4b_dino_bs256_120k"),
)

# Reuse the current 800M+DINO data, token, mask, and loss contracts, but restore
# the official Klein 4B pretrained backbone and ZeRO-2 launcher.
config["launch"] = copy.deepcopy(_klein4b_config["launch"])
config["launch"]["gpu_ids"] = [
    int(value)
    for value in os.environ.get("ROBONANA_GPU_IDS", "0,1,2,3,4,5,6,7").split(",")
    if value.strip()
]
config["dataloaders"]["train"]["batch_size_per_gpu"] = int(
    os.environ.get("ROBONANA_BATCH_SIZE", "16")
)

config["models"].update(
    initialization="pretrained",
    checkpoint=_klein4b_config["models"]["checkpoint"],
    checkpoint_dir=_klein4b_config["models"]["checkpoint_dir"],
    params=copy.deepcopy(_klein4b_config["models"]["params"]),
    train_mode="full",
    gradient_checkpointing=False,
    dino_dim=3072,
    dino_encoder_batch_size=int(
        os.environ.get("ROBONANA_DINO_ENCODER_BATCH_SIZE", "48")
    ),
)
config["optimizers"].update(
    lr=float(os.environ.get("ROBONANA_BACKBONE_LR", "2e-5")),
    robot_lr=float(os.environ.get("ROBONANA_ROBOT_LR", "1e-4")),
)
config["train"].update(
    gradient_accumulation_steps=2,
    pixel_eval_interval=int(os.environ.get("ROBONANA_PIXEL_EVAL_INTERVAL", "2000")),
)
