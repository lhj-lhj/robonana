"""中文：用原配置续训 Stage 1，关闭梯度检查点；保留 batch、精度和训练预算。

English: Resume saved Stage 1 without activation recomputation. This is a
configuration adapter, not another trainer. Use run_robotwin_train.sh with
--config robonana.configs.world_policy_resume.config and the same three
ROBONANA_RESUME_CHECKPOINT / RESUME_CONFIG / PROJECT_DIR variables as the
existing critic continuation. Original files remain untouched.
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
)
