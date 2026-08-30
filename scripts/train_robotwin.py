#!/usr/bin/env python3
"""Launch the RoboNana training config through FACT's launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fact_train import launch_from_config


DEFAULT_TRAIN_CONFIG = "robonana.configs.robotwin_flux2_4b_dino.config"
DEFAULT_PROJECT_DIR = "experiments/robotwin_flux2_4b_dino_bs256_120k"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_TRAIN_CONFIG)
    args = parser.parse_args()
    project_dir = Path(
        os.environ.get(
            "ROBONANA_PROJECT_DIR",
            DEFAULT_PROJECT_DIR,
        )
    ).resolve()
    os.environ.setdefault("WANDB_DIR", str(project_dir / "wandb"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(project_dir / "wandb" / "cache"))
    launch_from_config(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
