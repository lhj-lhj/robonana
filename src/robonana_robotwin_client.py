"""Thin compatibility wrapper around FACT's RoboTwin client adapter."""

from __future__ import annotations

import atexit
import os
import random
from pathlib import Path
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


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _response_scalar(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1)[0].item()
    return float(np.asarray(value).reshape(-1)[0])


def _install_chunk_return_hook(model) -> None:
    """Retain server Stage-2 outputs for the full environment action chunk."""

    if getattr(model, "_robonana_chunk_return_hook", False):
        return
    original_inference = model.client.inference

    def inference_with_chunk_return(request):
        response = original_inference(request)
        if isinstance(response, dict) and response.get("chunk_q") is not None:
            model._robonana_chunk_reward = _response_scalar(response["chunk_reward"])
            model._robonana_chunk_q = _response_scalar(response["chunk_q"])
            model._robonana_return_horizon = int(response.get("return_horizon", 0))
            model._robonana_chunk_index = int(
                getattr(model, "_robonana_chunk_index", -1)
            ) + 1
            image = response.get("images")
            if image is not None:
                model._robonana_pending_stage2_image = image
        return response

    model.client.inference = inference_with_chunk_return
    model._robonana_chunk_reward = None
    model._robonana_chunk_q = None
    model._robonana_return_horizon = 0
    model._robonana_chunk_index = -1
    model._robonana_pending_stage2_image = None
    model._robonana_chunk_return_hook = True


def _save_pending_stage2_image(task_env, model) -> Path | None:
    """Save one decoded Stage-2 future composite for each newly sampled action chunk."""

    image = getattr(model, "_robonana_pending_stage2_image", None)
    if image is None:
        return None
    model._robonana_pending_stage2_image = None
    output_root = os.environ.get("ROBONANA_STAGE2_IMAGE_ROOT", "").strip()
    if not output_root:
        return None

    if hasattr(image, "detach"):
        image = image.detach().cpu()
    array = np.asarray(image)
    if array.ndim != 5 or array.shape[0] != 1 or array.shape[1] != 3 or array.shape[2] != 1:
        raise ValueError(f"Stage-2 image must have shape [1,3,1,H,W], got {array.shape}")
    frame = np.transpose(array[0, :, 0], (1, 2, 0)).astype(np.float32, copy=False)
    frame = np.clip((frame + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)

    task_name = str(getattr(task_env, "task_name", "unknown_task"))
    episode_index = int(getattr(task_env, "test_num", 0))
    chunk_index = int(getattr(model, "_robonana_chunk_index", 0))
    horizon = int(getattr(model, "_robonana_return_horizon", 0))
    task_dir = Path(output_root).expanduser().resolve() / task_name / f"episode_{episode_index:06d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    output = task_dir / f"chunk_{chunk_index:03d}_h_{horizon:03d}.png"
    from PIL import Image

    Image.fromarray(frame, mode="RGB").save(output)
    return output


def _overlay_return(frame: np.ndarray, label: str) -> np.ndarray:
    import cv2

    output = np.ascontiguousarray(frame).copy()
    height, width = output.shape[:2]
    scale = max(0.45, width / 900.0)
    thickness = max(1, int(round(width / 640.0)))
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    x, y = 12, 12 + text_height
    cv2.rectangle(
        output,
        (x - 6, y - text_height - 6),
        (min(width - 1, x + text_width + 6), min(height - 1, y + baseline + 6)),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        output,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return output


class _ChunkReturnOverlayStream:
    """Annotate each raw RoboTwin RGB frame before it reaches ffmpeg."""

    def __init__(self, stream, task_env, model) -> None:
        self._stream = stream
        self._task_env = task_env
        self._model = model

    def write(self, data):
        reward = getattr(self._model, "_robonana_chunk_reward", None)
        q = getattr(self._model, "_robonana_chunk_q", None)
        if reward is None or q is None:
            return self._stream.write(data)
        try:
            reference = np.asarray(
                self._task_env.now_obs["observation"]["head_camera"]["rgb"]
            )
            if reference.ndim != 3 or reference.shape[-1] != 3:
                return self._stream.write(data)
            frame_bytes = int(reference.size)
            raw = bytes(data)
            if not raw or len(raw) % frame_bytes:
                return self._stream.write(data)
            horizon = int(getattr(self._model, "_robonana_return_horizon", 0))
            chunk_index = int(getattr(self._model, "_robonana_chunk_index", 0))
            label = (
                f"chunk={chunk_index:03d}  h={horizon}  "
                f"reward={float(reward):.4f}  Q={float(q):.4f}"
            )
            annotated = []
            for offset in range(0, len(raw), frame_bytes):
                frame = np.frombuffer(
                    raw[offset : offset + frame_bytes],
                    dtype=np.uint8,
                ).reshape(reference.shape)
                annotated.append(_overlay_return(frame, label).tobytes())
            return self._stream.write(b"".join(annotated))
        except Exception as error:
            if not getattr(self, "_warned", False):
                print(f"[RoboNana video] return overlay disabled for frame: {error}", flush=True)
                self._warned = True
            return self._stream.write(data)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install_video_return_overlay(task_env, model) -> None:
    if not _env_bool("ROBONANA_OVERLAY_CHUNK_RETURN", False):
        return
    ffmpeg = getattr(task_env, "eval_video_ffmpeg", None)
    if ffmpeg is None or isinstance(ffmpeg.stdin, _ChunkReturnOverlayStream):
        return
    ffmpeg.stdin = _ChunkReturnOverlayStream(ffmpeg.stdin, task_env, model)


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
        policy_version=os.environ.get(
            "ROBONANA_POLICY_VERSION",
            os.environ.get("ROBONANA_ROLLOUT_CHECKPOINT", str(usr_args.get("ckpt_setting", ""))),
        ),
        round_id=int(os.environ.get("ROBONANA_COLLECTION_ROUND", "0")),
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
    _install_chunk_return_hook(model)
    model._robonana_rollout_writer = writer
    if writer is not None:
        atexit.register(_finish_pending_rollout, model)
        print(f"[RoboNana rollout] recording to {writer.dataset_root}", flush=True)
    return model


def reset_model(model) -> None:
    _finish_pending_rollout(model)
    _fact_reset_model(model)
    model._robonana_chunk_reward = None
    model._robonana_chunk_q = None
    model._robonana_return_horizon = 0
    model._robonana_chunk_index = -1
    model._robonana_pending_stage2_image = None


def _episode_seed(task_env) -> int | None:
    for name in ("current_seed", "episode_seed", "seed", "now_seed"):
        if hasattr(task_env, name):
            return int(getattr(task_env, name))
    return None


def eval(TASK_ENV, model, observation):  # noqa: A001,N803
    _align_eval_instruction_with_training(TASK_ENV)
    _install_video_return_overlay(TASK_ENV, model)
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
        result = _fact_eval(TASK_ENV, model, observation)
        _save_pending_stage2_image(TASK_ENV, model)
        return result
    if observation.get("_fact_light_obs", False):
        observation = TASK_ENV._fact_force_full_obs()
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32).copy()
    images = {
        camera: np.asarray(observation["observation"][camera]["rgb"]).copy()
        for camera in CAMERAS
    }
    needed_new_plan = model.needs_new_plan(step)
    result = _fact_eval(TASK_ENV, model, observation)
    _save_pending_stage2_image(TASK_ENV, model)
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
    if terminal:
        final_observation = getattr(TASK_ENV, "now_obs", None)
        if final_observation is None:
            raise RuntimeError("RoboTwin terminal transition did not expose the final observation")
        if final_observation.get("_fact_light_obs", False):
            final_observation = TASK_ENV._fact_force_full_obs()
        final_state = np.asarray(
            final_observation["joint_action"]["vector"], dtype=np.float32
        ).copy()
        final_images = {
            camera: np.asarray(final_observation["observation"][camera]["rgb"]).copy()
            for camera in CAMERAS
        }
        writer.append_final_observation(images=final_images, state=final_state)
    return result

__all__ = ["eval", "get_model", "reset_model"]
