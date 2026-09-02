"""Fixed iterative-posttraining config shared by RoboNana model sizes."""

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
    failure_terminal_q: float,
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
        failure_terminal_q=failure_terminal_q,
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
    if pool_name == "historical_failure_replay":
        config["round_max"] = current_round - 1
    elif pool_name == "latest_failure":
        config["round_id"] = current_round
    return config


def apply_iterative_posttrain_config(config: dict[str, Any]) -> dict[str, Any]:
    """Convert one existing RoboNana train config into iterative posttraining."""

    config = copy.deepcopy(config)
    replay_root_value = os.environ.get("ROBONANA_REPLAY_ROOT", "").strip()
    checkpoint = os.environ.get("ROBONANA_POSTTRAIN_CHECKPOINT", "").strip()
    if not replay_root_value:
        raise ValueError("ROBONANA_REPLAY_ROOT is required for iterative posttraining")
    if not checkpoint:
        raise ValueError("ROBONANA_POSTTRAIN_CHECKPOINT is required for θ_k initialization")
    replay_root = Path(replay_root_value).expanduser()
    current_round = int(os.environ.get("ROBONANA_COLLECTION_ROUND", "0"))
    if current_round < 0:
        raise ValueError("ROBONANA_COLLECTION_ROUND cannot be negative")

    q_target_mode = os.environ.get(
        "ROBONANA_POSTTRAIN_Q_TARGET_MODE", "td_posttrain"
    ).strip()
    if q_target_mode not in {"td_posttrain", "mc_posttrain"}:
        raise ValueError(
            "ROBONANA_POSTTRAIN_Q_TARGET_MODE must be td_posttrain or mc_posttrain"
        )
    failure_terminal_q = float(
        os.environ.get("ROBONANA_FAILURE_TERMINAL_Q", "-1000.0")
    )
    mc_posttrain = q_target_mode == "mc_posttrain"

    original = copy.deepcopy(config["dataloaders"]["train"]["data_or_config"])
    if isinstance(original, list):
        original = original[0]
    original.update(
        q_target_mode=q_target_mode,
        failure_terminal_q=failure_terminal_q,
        episode_filter="success",
        pool_name="original_success",
        allow_empty=False,
        require_final_observation=False,
        discount=0.999,
        reward_non_goal=-1.0,
        reward_goal=0.0,
        dino_image_size=(480, 640) if original.get("dino_online", False) else None,
    )
    original_task_globs = os.environ.get(
        "ROBONANA_POSTTRAIN_ORIGINAL_TASK_GLOBS", ""
    ).strip()
    if original_task_globs:
        original["task_globs"] = tuple(
            value.strip() for value in original_task_globs.split(",") if value.strip()
        )
    stats_path = Path(
        os.environ.get("ROBONANA_REPLAY_STATS_PATH", str(original["stats_path"]))
    ).expanduser()
    dino_online = bool(original.get("dino_online", False))
    pools = [
        original,
        _replay_dataset_config(
            replay_root=replay_root,
            stats_path=stats_path,
            pool_name="collected_success_replay",
            episode_filter="success",
            current_round=current_round,
            dino_online=dino_online,
            q_target_mode=q_target_mode,
            failure_terminal_q=failure_terminal_q,
        ),
        _replay_dataset_config(
            replay_root=replay_root,
            stats_path=stats_path,
            pool_name="historical_failure_replay",
            episode_filter="failure",
            current_round=current_round,
            dino_online=dino_online,
            q_target_mode=q_target_mode,
            failure_terminal_q=failure_terminal_q,
        ),
        _replay_dataset_config(
            replay_root=replay_root,
            stats_path=stats_path,
            pool_name="latest_failure",
            episode_filter="failure",
            current_round=current_round,
            dino_online=dino_online,
            q_target_mode=q_target_mode,
            failure_terminal_q=failure_terminal_q,
        ),
    ]
    config["dataloaders"]["train"]["data_or_config"] = pools
    pool_weights = (
        dict(
            original_success=5.0 / 11.0,
            collected_success_replay=1.0 / 22.0,
            historical_failure_replay=0.0,
            latest_failure=0.5,
        )
        if mc_posttrain
        else dict(
            original_success=0.25,
            collected_success_replay=0.25,
            historical_failure_replay=0.25,
            latest_failure=0.25,
        )
    )
    config["dataloaders"]["train"]["sampler"] = dict(
        type="RoboTwinPosttrainSampler",
        infinite=True,
        pool_weights=pool_weights,
        redistribute_empty_historical_failure_to_latest=True,
        redistribute_empty_collected_success_to_original=True,
    )
    config["models"].update(
        initialization="trained",
        checkpoint=checkpoint,
    )
    checkpoint_config = os.environ.get("ROBONANA_POSTTRAIN_MODEL_CONFIG", "").strip()
    if checkpoint_config:
        config["models"]["checkpoint_config"] = checkpoint_config

    posttrain = dict(
        enabled=True,
        discount=0.999,
        reward_non_goal=-1.0,
        reward_goal=0.0,
        reward_normalization="none",
        q_normalization="none",
        q_target_mode=q_target_mode,
        failure_terminal_q=failure_terminal_q,
        current_collection_round=current_round,
        ema=dict(
            decay=0.995,
            update_every_optimizer_steps=1,
            start_step=0,
            storage_dtype="float32",
            forward_autocast_dtype="bfloat16",
            include_full_flux_fact_model=True,
            include_qwen=False,
            include_vae=False,
            include_dino_encoder=False,
        ),
        data_mixture=dict(
            **pool_weights,
            redistribute_empty_historical_failure_to_latest=True,
            redistribute_empty_collected_success_to_original=True,
            task_balanced=True,
            episode_balanced_within_task=True,
        ),
        failure_policy_improvement=dict(
            apply_to_historical_failures=True,
            apply_to_latest_failures=True,
            candidate_policy="online",
            candidate_count=8,
            candidate_horizon=48,
            candidate_action_sampling_steps=20,
            candidate_q_sampling_steps=20,
            flow_shift=1.0,
            candidate_microbatch_size=int(
                os.environ.get("ROBONANA_CANDIDATE_MICROBATCH_SIZE", "16")
            ),
            common_world_noise_across_candidates=True,
            candidate_selection="argmax",
            use_behavior_candidate=False,
            use_advantage_gate=False,
            use_confidence_gate=False,
            use_uncertainty_gate=False,
            pseudo_action_weight=1.0,
        ),
        td=dict(
            next_action_policy="ema",
            target_q_model="ema",
            target_action_horizon=48,
            target_action_samples=1,
            action_sampling_steps=20,
            q_sampling_steps=20,
            flow_shift=1.0,
            bootstrap_success_terminal=False,
            bootstrap_failure_timeout=True,
            stop_gradient=True,
            microbatch_size=int(os.environ.get("ROBONANA_TD_MICROBATCH_SIZE", "16")),
        ),
    )
    config["train"].update(
        posttrain=posttrain,
        discount=0.999,
        reward_non_goal=-1.0,
        reward_goal=0.0,
        q_target_mode=q_target_mode,
        failure_terminal_q=failure_terminal_q,
        checkpoint_save_optimizer=True,
        resume=_env_flag("ROBONANA_RESUME", True),
    )
    config["train"]["loss_weights"].update(
        image_loss=1.0,
        action_loss=10.0,
        future_state_loss=0.4,
        reward_loss=0.01,
        success_loss=0.01,
        q_loss=0.001,
    )
    config["project_dir"] = os.environ.get(
        "ROBONANA_PROJECT_DIR",
        str(Path(config["project_dir"]).with_name(Path(config["project_dir"]).name + "_posttrain")),
    )
    return config
