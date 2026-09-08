#!/usr/bin/env python3
"""Bounded real-VAE cache/live parity probe; never modifies dataset caches.

Use an existing HDF5 episode and optionally one LeRobot task. Tests identical
decoded pixels through offline and online entry points, including batch=2.
This cannot undo historical JPEG/MP4 loss or certify old training caches.
"""
import argparse
import json
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from diffusers.models import AutoencoderKLFlux2

from preprocess_robotwin_flux import build_composite_batch, HDF5_CAMERAS
from robonana.encoding import encode_flux2_image_tokens
from robonana.image_pipeline import (
    VIEW_KEYS, build_robotwin_vae_input, image_contract,
)
from robonana.inference.batched_policy import BatchedRoboNanaRobotWinPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--lerobot-task", type=Path)
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()
    vae = AutoencoderKLFlux2.from_pretrained(
        args.checkpoint, subfolder="vae", torch_dtype=torch.float32,
        local_files_only=True).eval().requires_grad_(False).to(args.device)
    observations = []
    with h5py.File(args.hdf5) as handle:
        offline = build_composite_batch(handle, 0, 2)
        for frame in range(2):
            obs = {}
            for key, camera in zip(VIEW_KEYS, HDF5_CAMERAS, strict=True):
                with Image.open(BytesIO(bytes(handle[f"observation/{camera}/rgb"][frame]))) as image:
                    obs[key] = np.asarray(image.convert("RGB")).copy()
            observations.append(obs)
    report = {"contract": image_contract(str(args.checkpoint)), "sources": {}}
    # Exercise the actual live adapters without loading unrelated FLUX/Qwen
    # weights. These two image methods require only the frozen VAE and layout.
    policy = object.__new__(BatchedRoboNanaRobotWinPolicy)
    policy.vae = vae
    policy.main_view_size = (256, 192)
    policy.grid_height, policy.grid_width = 12, 24
    policy.model_device, policy.dtype = torch.device(args.device), torch.float32

    def check(label, pixels, rows):
        live_pixels = torch.cat([build_robotwin_vae_input(row) for row in rows])
        assert torch.equal(pixels, live_pixels), f"{label}: pixel mismatch"
        cache = encode_flux2_image_tokens(vae, pixels.to(args.device)).bfloat16().float()
        batch = policy._batched_current_image_tokens(rows)
        solo = torch.cat([policy._current_image_tokens(row) for row in rows])
        assert torch.equal(cache, batch) and torch.equal(cache, solo), f"{label}: token mismatch"
        report["sources"][label] = {"pixel_equal": True, "cache_live_b1_b2_equal": True,
                                     "max_abs_error": float((cache - batch).abs().max()),
                                     "shape": list(cache.shape)}

    check("hdf5_replay", offline, observations)
    if args.lerobot_task:
        from preprocess_robotwin_lerobot_flux import decode_views
        # Decoder verifies full episode indexing; retain only two decoded frames
        # for the expensive VAE test.
        rows = [json.loads(line) for line in (args.lerobot_task / "meta/episodes.jsonl").read_text().splitlines()]
        row = rows[0]
        views = decode_views(args.lerobot_task, row["episode_index"], row["length"], 1)
        raw = {key: value[:2] for key, value in views.items()}
        obs = [{key: value[i] for key, value in raw.items()} for i in range(2)]
        check("lerobot_original", build_robotwin_vae_input(raw), obs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
