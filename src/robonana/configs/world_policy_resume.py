"""中文：用原配置续训 Stage 1，配置激活重计算；保留 batch、精度和训练预算。

English: Resume saved Stage 1 with explicit activation recomputation. This is a
configuration adapter, not another trainer. Use run_robotwin_train.sh with
--config robonana.configs.world_policy_resume.config and the same three
ROBONANA_RESUME_CHECKPOINT / RESUME_CONFIG / PROJECT_DIR variables as the
existing critic continuation. Original files remain untouched.

ROBONANA_ADDITIONAL_STEPS extends a completed run with the same peak LR and
warmup using FACT's original cosine module; optimizer/global progress resume.

中文：部分重计算设置 ROBONANA_GRADIENT_CHECKPOINTING=1 和
ROBONANA_GRADIENT_CHECKPOINTING_SINGLE_STRIDE=2；保留全部double和偶数single的检查点。
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
    additional_steps=int(os.environ.get("ROBONANA_ADDITIONAL_STEPS", "0")),
    gpu_ids=tuple(int(value) for value in os.environ["ROBONANA_RESUME_GPUS"].split(","))
        if "ROBONANA_RESUME_GPUS" in os.environ else None,
    batch_size_per_gpu=int(os.environ["ROBONANA_RESUME_BATCH_SIZE_PER_GPU"])
        if "ROBONANA_RESUME_BATCH_SIZE_PER_GPU" in os.environ else None,
    accumulation_steps=int(os.environ["ROBONANA_RESUME_ACCUMULATION"])
        if "ROBONANA_RESUME_ACCUMULATION" in os.environ else None,
    global_batch=int(os.environ["ROBONANA_RESUME_GLOBAL_BATCH"])
        if "ROBONANA_RESUME_GLOBAL_BATCH" in os.environ else None,
)
if os.environ.get("ROBONANA_UNIVERSAL_CHECKPOINT") == "1":
    checkpoint = Path(os.environ["ROBONANA_RESUME_CHECKPOINT"]).resolve()
    ds_config = checkpoint / "deepspeed_universal.json"
    if not ds_config.is_file():
        raise FileNotFoundError(ds_config)
    config["launch"]["deepspeed_config"] = {"deepspeed_config_file": str(ds_config)}
