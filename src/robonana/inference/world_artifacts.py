"""Small NumPy/Pillow writer usable in the separate RoboTwin environment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def save_selected_world(root, *, task, seed, step, request, response):
    """One folder per seed; use the pre-execution control step as chunk identity."""
    world = dict(response["selected_world"])
    image = _array(world.pop("image"))
    if image.ndim != 5 or image.shape[:3] != (1, 3, 1):
        raise ValueError(f"expected selected future [1,3,1,H,W], got {image.shape}")
    directory = Path(root) / task / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"step_{step:04d}"
    frame = np.transpose(image[0, :, 0], (1, 2, 0))
    Image.fromarray(np.clip((frame + 1) * 127.5, 0, 255).astype(np.uint8)).save(
        directory / f"{stem}_predicted_t48.png"
    )
    for key in ("observation.images.cam_high", "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist"):
        if key not in request:
            continue
        frame = _array(request[key])
        if frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        Image.fromarray(frame).save(directory / f"{stem}_{key.rsplit('.', 1)[-1]}.png")
    world.update(
        task=task, seed=seed, step=step, sampling_seed=response.get("_sampling_seed"),
        selected_q=response["selected_q"],
        selected_candidate_index=response["selected_candidate_index"],
        candidate_q=_array(response["candidate_q"]).tolist(),
        action=_array(response["action"]).tolist(),
        instruction=request.get("instruction", request.get("prompt", "")),
        image=f"{stem}_predicted_t48.png",
        timing_ms=response.get("_policy_timing_ms", {}),
    )
    (directory / f"{stem}.json").write_text(json.dumps(world, indent=2, allow_nan=False), encoding="utf-8")
    return directory
