"""Thin compatibility wrapper around FACT's RoboTwin client adapter."""

from __future__ import annotations

import atexit
import os
import random
from types import MethodType

import numpy as np


def _patch_warp_torch_namespace() -> None:
    """Restore the namespace expected by CuRobo 0.7.8 under Warp 1.16+."""

    try:
        import warp
    except ImportError:
        return
    if hasattr(warp, "torch"):
        return
    try:
        from warp._src import torch as warp_torch
    except ImportError:
        return
    warp.torch = warp_torch


_patch_warp_torch_namespace()

from evaluation.robotwin.model2robotwin_interface import (  # noqa: E402
    eval as _fact_eval,
    get_model as _fact_get_model,
    reset_model as _fact_reset_model,
)
from robonana.data.rollout_writer import CAMERAS, RoboTwinRolloutWriter  # noqa: E402


def _seed_python_random(seed: int) -> None:
    """Seed the RNG RoboTwin uses for instruction shuffle/selection."""

    random.seed(int(seed))


def _patch_robotwin_python_random_seed() -> None:
    """Make a RoboTwin episode seed cover Python-random instruction generation."""

    try:
        from envs._base_task import Base_Task
    except ImportError:
        return
    if getattr(Base_Task, "_robonana_python_random_seed_patch", False):
        return
    original_init_task_env = Base_Task._init_task_env_

    def _init_task_env_with_python_seed(self, *args, **kwargs):
        if "seed" in kwargs:
            _seed_python_random(int(kwargs["seed"]))
        return original_init_task_env(self, *args, **kwargs)

    Base_Task._init_task_env_ = _init_task_env_with_python_seed
    Base_Task._robonana_python_random_seed_patch = True


def _align_eval_instruction_with_training(task_env) -> str:
    """Optionally replace RoboTwin's sampled paraphrase with the training prompt."""

    instruction = os.environ.get("ROBONANA_EVAL_INSTRUCTION", "").strip()
    if instruction:
        task_env.set_instruction(instruction=instruction)
    return str(task_env.get_instruction())


def sampling_seed_for_step(episode_seed: int, step: int, execute_actions_per_plan: int) -> int:
    """Return a stable diffusion seed for one episode/replanning point."""

    interval = max(1, int(execute_actions_per_plan))
    plan_index = int(step) // interval
    return int(episode_seed) * 1_000_003 + plan_index


def _install_sampling_seed_hook(model) -> None:
    """Forward the tracked RoboTwin episode seed through FACT's request builder."""

    if getattr(model, "_robonana_sampling_seed_hook", False):
        return
    original_build_request = model._build_request

    def _build_request_with_seed(self, example):
        request = original_build_request(example)
        sampling_seed = getattr(self, "_robonana_sampling_seed", None)
        if sampling_seed is not None:
            request["sampling_seed"] = int(sampling_seed)
        return request

    model._build_request = MethodType(_build_request_with_seed, model)
    model._robonana_sampling_seed_hook = True


def _rollout_writer(usr_args) -> RoboTwinRolloutWriter | None:
    dataset_root = os.environ.get("ROBONANA_ROLLOUT_DATASET_ROOT", "").strip()
    if not dataset_root:
        return None
    return RoboTwinRolloutWriter(
        dataset_root,
        initial_dataset_root=os.environ.get(
            "ROBONANA_INITIAL_DATASET_ROOT",
            "/workspace/datasets/RoboTwin/hf_dataset",
        ),
        jpeg_quality=int(os.environ.get("ROBONANA_ROLLOUT_JPEG_QUALITY", "95")),
        policy_name=str(usr_args.get("policy_name", "robonana_robotwin.adapter")),
        checkpoint=os.environ.get(
            "ROBONANA_ROLLOUT_CHECKPOINT",
            str(usr_args.get("ckpt_setting", "")),
        ),
        task_config=str(usr_args.get("task_config", "")),
    )


def _finish_pending_rollout(model) -> None:
    writer = getattr(model, "_robonana_rollout_writer", None)
    if writer is None:
        return
    output = writer.finish_episode()
    if output is not None:
        print(f"[RoboNana rollout] saved {output}", flush=True)


def get_model(usr_args):
    _patch_robotwin_python_random_seed()
    writer = _rollout_writer(usr_args)
    fact_args = dict(usr_args)
    if writer is not None:
        # Training rollouts need RGB at every control step, not just at replans.
        fact_args["low_frequency_rgb"] = False
        fact_args["skip_action_render_sync"] = False
    model = _fact_get_model(fact_args)
    _install_sampling_seed_hook(model)
    model._robonana_rollout_writer = writer
    if writer is not None:
        atexit.register(_finish_pending_rollout, model)
        print(f"[RoboNana rollout] recording to {writer.dataset_root}", flush=True)
    return model


def reset_model(model) -> None:
    _finish_pending_rollout(model)
    _fact_reset_model(model)


def _episode_seed(task_env) -> int | None:
    for name in ("current_seed", "episode_seed", "seed", "now_seed"):
        if hasattr(task_env, name):
            return int(getattr(task_env, name))
    return None


def eval(TASK_ENV, model, observation):  # noqa: A001,N803
    _align_eval_instruction_with_training(TASK_ENV)
    step = int(getattr(TASK_ENV, "take_action_cnt", 0))
    episode_seed = _episode_seed(TASK_ENV)
    if episode_seed is not None and model.needs_new_plan(step):
        model._robonana_sampling_seed = sampling_seed_for_step(
            episode_seed,
            step,
            model.execute_actions_per_plan,
        )
    writer = getattr(model, "_robonana_rollout_writer", None)
    if writer is None:
        return _fact_eval(TASK_ENV, model, observation)
    if observation.get("_fact_light_obs", False):
        observation = TASK_ENV._fact_force_full_obs()
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32).copy()
    images = {
        camera: np.asarray(observation["observation"][camera]["rgb"]).copy()
        for camera in CAMERAS
    }
    needed_new_plan = model.needs_new_plan(step)
    result = _fact_eval(TASK_ENV, model, observation)
    plan_offset = 0 if needed_new_plan else step % model.execute_actions_per_plan
    action = np.asarray(model.planned_actions[plan_offset], dtype=np.float32).copy()
    success = bool(getattr(TASK_ENV, "eval_success", False))
    terminal = success or int(getattr(TASK_ENV, "take_action_cnt", 0)) >= int(TASK_ENV.step_lim)
    writer.append(
        task_name=str(getattr(TASK_ENV, "task_name", "unknown_task")),
        instruction=str(TASK_ENV.get_instruction()),
        seed=_episode_seed(TASK_ENV),
        images=images,
        state=state,
        action=action,
        success=success,
        terminal=terminal,
    )
    return result

__all__ = ["eval", "get_model", "reset_model"]
