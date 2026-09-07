"""Resume a saved critic phase through the single maintained RoboNana trainer."""

import json
import os
from pathlib import Path

from robonana.data import robotwin_lerobot as _register_lerobot  # noqa: F401
from robonana.training.continuation import build_critic_continuation

checkpoint = Path(os.environ["ROBONANA_RESUME_CHECKPOINT"]).resolve()
source_config = Path(os.environ["ROBONANA_RESUME_CONFIG"]).resolve()
project = Path(os.environ["ROBONANA_PROJECT_DIR"]).resolve()
if project == source_config.parent or project in checkpoint.parents:
    raise ValueError("use a separate project directory; preserve the source experiment")
config = build_critic_continuation(
    json.loads(source_config.read_text()), checkpoint=checkpoint,
    source_config=source_config, project_dir=project,
    max_steps=int(os.environ.get("ROBONANA_MAX_STEPS", "10000")),
    batch_size_per_gpu=int(os.environ.get("ROBONANA_BATCH_SIZE_PER_GPU", "8")),
)
