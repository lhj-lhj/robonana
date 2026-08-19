"""RoboTwin 2.0 full-training config for FACT's loop and FLUX.2 Klein 4B."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = Path(os.environ.get("ROBONANA_DATASET_ROOT", "/workspace/datasets/RoboTwin/hf_dataset"))
CHECKPOINT_DIR = Path(
    os.environ.get("ROBONANA_FLUX_CHECKPOINT_DIR", REPO_ROOT / "checkpoints" / "FLUX.2-klein-base-4B")
)
BACKBONE_CHECKPOINT = Path(
    os.environ.get(
        "ROBONANA_FLUX_BACKBONE",
        CHECKPOINT_DIR / "flux-2-klein-base-4b.safetensors",
    )
)
PROJECT_DIR = os.environ.get("ROBONANA_PROJECT_DIR", str(REPO_ROOT / "experiments" / "robotwin_flux2"))
GPU_IDS = [int(value) for value in os.environ.get("ROBONANA_GPU_IDS", "0,2,5,7").split(",") if value.strip()]
MAX_STEPS = int(os.environ.get("ROBONANA_MAX_STEPS", "150000"))
BATCH_SIZE_PER_GPU = int(os.environ.get("ROBONANA_BATCH_SIZE", "1"))
NUM_WORKERS = int(os.environ.get("ROBONANA_NUM_WORKERS", "4"))
TRAIN_MODE = os.environ.get("ROBONANA_TRAIN_MODE", "full")
PIXEL_EVAL_INTERVAL = int(os.environ.get("ROBONANA_PIXEL_EVAL_INTERVAL", "200"))
LOG_INTERVAL = int(os.environ.get("ROBONANA_LOG_INTERVAL", "10"))
MEMORY_LIMIT_GIB = float(os.environ.get("ROBONANA_MEMORY_LIMIT_GIB", "0"))
NUM_INFERENCE_STEPS = int(os.environ.get("ROBONANA_NUM_INFERENCE_STEPS", "20"))
EARLY_CHECKPOINT_STEPS = tuple(
    int(value)
    for value in os.environ.get("ROBONANA_EARLY_CHECKPOINT_STEPS", "100").split(",")
    if value.strip()
)
DISABLE_CHECKPOINTING = os.environ.get("ROBONANA_DISABLE_CHECKPOINTING", "0").lower() in {
    "1",
    "true",
    "yes",
}

DEEPSPEED_CONFIG = (
    REPO_ROOT / "third_party" / "FACT" / "fact_train" / "distributed" / "accelerate_configs" / "zero2.json"
)

config = dict(
    project_dir=PROJECT_DIR,
    runners=["robonana.training.robotwin_trainer.RoboNanaTrainer"],
    launch=dict(
        gpu_ids=GPU_IDS,
        distributed_type="DEEPSPEED",
        deepspeed_config=dict(deepspeed_config_file=str(DEEPSPEED_CONFIG)),
        executable=f"{sys.executable} -m accelerate.commands.accelerate_cli",
        until_completion=False,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=dict(
                _class_name="RoboTwinHDF5Dataset",
                data_path=str(DATASET_ROOT),
                stats_path=str(DATASET_ROOT / "robonana_norm_stats.json"),
                index_path=str(DATASET_ROOT / "robonana_index.json"),
                task_glob="*/aloha-agilex_clean_50",
                action_chunk=48,
                action_dim=14,
                max_horizon=48,
                eval_horizons=(12, 24, 48),
            ),
            batch_size_per_gpu=BATCH_SIZE_PER_GPU,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=4 if NUM_WORKERS > 0 else None,
            transform=None,
            sampler=dict(type="RoboTwinEpisodeSampler", infinite=True),
            collator=dict(is_equal=True),
        ),
        test=dict(),
    ),
    models=dict(
        checkpoint=str(BACKBONE_CHECKPOINT),
        checkpoint_dir=str(CHECKPOINT_DIR),
        action_dim=14,
        state_dim=14,
        max_horizon=48,
        train_mode=TRAIN_MODE,
        gradient_checkpointing=True,
        vae_dtype="float32",
    ),
    optimizers=dict(
        type="AdamW",
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-4,
        fused=True,
        foreach=False,
    ),
    schedulers=dict(
        type="WarmupCosineScheduler",
        warmup_steps=500,
        decay_steps=MAX_STEPS,
    ),
    train=dict(
        max_steps=MAX_STEPS,
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        activation_checkpointing=False,
        checkpoint_interval=1000,
        early_checkpoint_steps=EARLY_CHECKPOINT_STEPS,
        checkpoint_total_limit=1,
        checkpoint_save_optimizer=False,
        disable_checkpointing=DISABLE_CHECKPOINTING,
        resume=True,
        log_with="wandb",
        tracker_project_name="robonana",
        tracker_init_kwargs=dict(wandb=dict(entity="hongjia-liu-aalto-university")),
        log_interval=LOG_INTERVAL,
        pixel_eval_interval=PIXEL_EVAL_INTERVAL,
        latent_grid_height=12,
        latent_grid_width=24,
        flow_shift=1.0,
        num_inference_steps=NUM_INFERENCE_STEPS,
        max_grad_norm=1.0,
        memory_limit_gib=MEMORY_LIMIT_GIB,
        loss_weights=dict(
            image_loss=1.0,
            action_loss=10.0,
            future_state_loss=0.4,
            value_loss=0.4,
        ),
    ),
)
