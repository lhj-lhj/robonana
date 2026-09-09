"""中文：用原配置续训 Stage 1，配置激活重计算；保留 batch、精度和训练预算。

English: Resume saved Stage 1 with explicit activation recomputation. This is a
configuration adapter, not another trainer. Use run_robotwin_train.sh with
--config robonana.configs.world_policy_resume.config and the same three
ROBONANA_RESUME_CHECKPOINT / RESUME_CONFIG / PROJECT_DIR variables as the
existing critic continuation. Original files remain untouched.

中文：部分重计算设置 GRADIENT_CHECKPOINTING=1、SINGLE_STRIDE=2（均加
ROBONANA_ 前缀，后者完整名见下）；保留全部double和偶数single的检查点。
English: Enable checkpointing and set single stride 2 to checkpoint all double
blocks and even single blocks; no tensor/layout/optimizer changes.
"""

import json
import os
from pathlib import Path

from robonana.data import robotwin_lerobot as _register_lerobot  # noqa: F401
from robonana.training.continuation import build_world_policy_resume

config = build_world_policy_resume(
    json.loads(Path(os.environ["ROBONANA_RESUME_CONFIG"]).read_text()),
    checkpoint=Path(os.environ["ROBONANA_RESUME_CHECKPOINT"]).resolve(),
    source_config=Path(os.environ["ROBONANA_RESUME_CONFIG"]).resolve(),
    project_dir=Path(os.environ["ROBONANA_PROJECT_DIR"]).resolve(),
    gradient_checkpointing=os.environ.get("ROBONANA_GRADIENT_CHECKPOINTING", "0") == "1",
    single_checkpoint_stride=int(os.environ.get("ROBONANA_GRADIENT_CHECKPOINTING_SINGLE_STRIDE", "1")),
)
