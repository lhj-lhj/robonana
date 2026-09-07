"""Maintained fixed-48 MAC two-phase training configuration."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _replay_dataset_config(
    *,
    replay_root: Path,
    stats_path: Path,
    pool_name: str,
    episode_filter: str,
    current_round: int,
    dino_online: bool,
    q_target_mode: str,
) -> dict[str, Any]:
    config: dict[str, Any] = dict(
        _class_name="RoboTwinHDF5Dataset",
        data_path=str(replay_root),
        stats_path=str(stats_path),
        index_path=str(replay_root / "robonana_index.json"),
        task_glob=os.environ.get("ROBONANA_REPLAY_TASK_GLOB", "**/robonana_rollout"),
        action_chunk=48,
        action_dim=14,
        max_horizon=48,
        eval_horizons=(12, 24, 48),
        discount=0.999,
        reward_non_goal=-1.0,
        reward_goal=0.0,
        q_target_mode=q_target_mode,
        episode_filter=episode_filter,
        pool_name=pool_name,
        dino_online=dino_online,
        dino_image_size=(480, 640) if dino_online else None,
        allow_empty=pool_name in {
            "collected_success_replay",
            "historical_failure_replay",
        },
        require_final_observation=True,
    )
    if q_target_mode == "mac_mot_v2":
        config["fixed_horizon"] = 48
    if pool_name == "historical_failure_replay":
        config["round_max"] = current_round - 1
    elif pool_name == "latest_failure":
        config["round_id"] = current_round
    return config

def apply_mac_posttrain_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build one fixed-48/H=1 MAC phase from a trained MAC checkpoint."""

    config = copy.deepcopy(config)
    config.setdefault("optimizers", {})
    repo_root = Path(__file__).resolve().parents[3]
    source_run = Path(os.environ.get(
        "ROBONANA_MAC_SOURCE_RUN",
        "/data3/hongjia/robonana/experiments/hanging_mug_mac_pilot_20260906/world_policy",
    )).expanduser()
    source_checkpoint = Path(
        os.environ.get(
            "ROBONANA_MAC_PRETRAIN_CHECKPOINT",
            source_run / "models/checkpoint_epoch_1_step_1000/transformer/diffusion_pytorch_model.bin",
        )
    ).expanduser()
    source_config = Path(
        os.environ.get("ROBONANA_MAC_PRETRAIN_CONFIG", source_run / "config.json")
    ).expanduser()
    replay_root = Path(
        os.environ.get(
            "ROBONANA_REPLAY_ROOT",
            "/data3/hongjia/robonana_rollouts/hanging_mug_round0_from160k",
        )
    ).expanduser()
    current_round = int(os.environ.get("ROBONANA_COLLECTION_ROUND", "0"))
    if current_round < 0:
        raise ValueError("ROBONANA_COLLECTION_ROUND cannot be negative")
    phase = os.environ.get("ROBONANA_MAC_PHASE", "world_policy").strip()
    if phase not in {"world_policy", "critic"}:
        raise ValueError("ROBONANA_MAC_PHASE must be world_policy or critic")
    # Budgets are per new phase/round, not lifetime checkpoint step counts.
    # Override the common base's long pretraining budget and its scheduler
    # together; otherwise a 10k critic run could inherit a 150k decay schedule.
    phase_steps = (
        os.environ.get("ROBONANA_MAC_WORLD_POLICY_STEPS",
                       os.environ.get("ROBONANA_MAC_TRAIN_STEPS", "20000"))
        if phase == "world_policy"
        else os.environ.get("ROBONANA_MAC_CRITIC_STEPS", "10000")
    )
    max_steps = int(os.environ.get("ROBONANA_MAX_STEPS", phase_steps))
    if max_steps <= 0:
        raise ValueError("MAC phase max_steps must be positive")
    config["train"]["max_steps"] = max_steps
    config.setdefault("schedulers", {})["decay_steps"] = max_steps

    original = copy.deepcopy(config["dataloaders"]["train"]["data_or_config"])
    if isinstance(original, list):
        original = original[0]
    original.update(
        q_target_mode="mac_mot_v2",
        fixed_horizon=48,
        max_horizon=48,
        action_chunk=48,
        episode_filter="success",
        pool_name="original_success",
        allow_empty=False,
        require_final_observation=False,
        dino_online=False,
        dino_image_size=None,
        task_globs=tuple(
            item.strip()
            for item in os.environ.get(
                "ROBONANA_POSTTRAIN_ORIGINAL_TASK_GLOBS", "Clean/hanging_mug"
            ).split(",")
            if item.strip()
        ),
    )
    stats_path = Path(
        os.environ.get("ROBONANA_REPLAY_STATS_PATH", str(original["stats_path"]))
    ).expanduser()
    pools = [
        original,
        _replay_dataset_config(
            replay_root=replay_root,
            stats_path=stats_path,
            pool_name="collected_success_replay",
            episode_filter="success",
            current_round=current_round,
            dino_online=False,
            q_target_mode="mac_mot_v2",
        ),
        _replay_dataset_config(
            replay_root=replay_root,
            stats_path=stats_path,
            pool_name="historical_failure_replay",
            episode_filter="failure",
            current_round=current_round,
            dino_online=False,
            q_target_mode="mac_mot_v2",
        ),
        _replay_dataset_config(
            replay_root=replay_root,
            stats_path=stats_path,
            pool_name="latest_failure",
            episode_filter="failure",
            current_round=current_round,
            dino_online=False,
            q_target_mode="mac_mot_v2",
        ),
    ]
    pool_weights = dict(
        original_success=0.25,
        collected_success_replay=0.25,
        historical_failure_replay=0.25,
        latest_failure=0.25,
    )
    config["dataloaders"]["train"].update(
        data_or_config=pools,
        sampler=dict(
            type="RoboTwinPosttrainSampler",
            infinite=True,
            pool_weights=pool_weights,
            redistribute_empty_historical_failure_to_latest=True,
            redistribute_empty_collected_success_to_original=True,
        ),
    )
    config["models"].update(
        architecture_version="mac_mot_v2",
        initialization="trained",
        checkpoint=str(source_checkpoint),
        checkpoint_config=str(source_config),
        action_dim=14,
        state_dim=14,
        chunk_horizon=48,
        max_horizon=48,
        reward_dim=48,
        reward_head_type="binary_chunk",
        success_dim=1,
        q_dim=1,
        value_dim=1,
        pred_action_bidirectional=True,
        dino_dim=None,
        expert_hidden_dim=int(os.environ.get("ROBONANA_MAC_EXPERT_HIDDEN_DIM", "1024")),
        train_mode=phase,
    )
    candidate_count = int(os.environ.get("ROBONANA_MAC_TRAIN_CANDIDATES", "8"))
    if candidate_count <= 0:
        raise ValueError("ROBONANA_MAC_TRAIN_CANDIDATES must be positive")
    posttrain = dict(
        enabled=True,
        algorithm="mac_mot_v2",
        q_target_mode="mac_mot_v2",
        phase=phase,
        chunk_horizon=48,
        discount=float(os.environ.get("ROBONANA_DISCOUNT", "0.999")),
        reward_non_goal=-1.0,
        reward_goal=0.0,
        return_scale=float(os.environ.get("ROBONANA_MAC_RETURN_SCALE", "1000.0")),
        current_collection_round=current_round,
        ema=dict(
            decay=0.995,
            update_every_optimizer_steps=1,
            start_step=0,
            storage_dtype="float32",
            target="value_expert_only",
        ),
        imagination=dict(
            rollout_chunks=1,
            candidate_count=candidate_count,
            sampling_steps=int(os.environ.get("ROBONANA_MAC_SAMPLING_STEPS", "20")),
            flow_shift=float(os.environ.get("ROBONANA_MAC_FLOW_SHIFT", "1.0")),
            candidate_selection="argmax_q",
            fresh_each_batch=True,
            stop_gradient_target=True,
        ),
        data_mixture=dict(
            **pool_weights,
            success_only_action_bc=True,
            all_real_rollouts_train_world=True,
            redistribute_empty_historical_failure_to_latest=True,
            redistribute_empty_collected_success_to_original=True,
        ),
        environment_policy=dict(
            candidate_count=int(os.environ.get("ROBONANA_MAC_EVAL_CANDIDATES", "32")),
            candidate_selection="argmax_q",
            action_chunk=48,
            execute_actions_per_plan=48,
        ),
    )
    config["train"].update(
        posttrain=posttrain,
        q_target_mode="mac_mot_v2",
        discount=posttrain["discount"],
        reward_non_goal=-1.0,
        reward_goal=0.0,
        resume=_env_flag("ROBONANA_RESUME", True),
        pixel_eval_interval=0,
        checkpoint_save_optimizer=True,
        with_ema=False,
    )
    config["train"]["loss_weights"].update(
        image_loss=1.0,
        action_loss=10.0,
        future_state_loss=0.4,
        reward_loss=0.1,
        success_loss=0.1,
        value_loss=1.0,
        q_loss=1.0,
        dino_loss=0.0,
    )
    if phase == "critic":
        critic_lr = float(os.environ.get("ROBONANA_MAC_CRITIC_LR", "1e-4"))
        config["optimizers"].update(lr=critic_lr, robot_lr=critic_lr)
    config["project_dir"] = os.environ.get(
        "ROBONANA_PROJECT_DIR",
        str(repo_root / "experiments" / "hanging_mug_mac"),
    )
    config["train"]["tracker_init_kwargs"]["wandb"].update(
        name=os.environ.get("WANDB_NAME", "hanging-mug-mac")
    )
    return config
