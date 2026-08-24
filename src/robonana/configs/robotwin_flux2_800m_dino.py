"""Scratch ~800M RoboNana with a trailing 147x3072 DINOv3 flow target."""

from __future__ import annotations

import copy
import os
from pathlib import Path

from .robotwin_flux2_800m import config as _base_config


config = copy.deepcopy(_base_config)
repo_root = Path(__file__).resolve().parents[3]
config["project_dir"] = os.environ.get(
    "ROBONANA_PROJECT_DIR",
    str(repo_root / "experiments" / "robotwin_flux2_800m_dino_full_bs256_120k"),
)
config["models"]["dino_dim"] = 3072
config["dataloaders"]["train"]["data_or_config"].update(
    dino_cache=True,
    dino_cache_size=1,
    dino_token_count=147,
    dino_feature_dim=3072,
)
config["train"]["loss_weights"]["dino_loss"] = float(
    os.environ.get("ROBONANA_DINO_LOSS_WEIGHT", "0.1")
)
