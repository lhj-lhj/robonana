#!/usr/bin/env python3
# 中文：诊断：验证缓存生成与在线 VAE 编码的一致性。
# English: Diagnostic: verify parity between cache generation and online VAE encoding.
# 调用 / Invocation: 显式真实 VAE 测试；不覆盖数据集缓存。 / Explicit real-VAE test; never overwrites dataset caches.
# 导航 / Guide: scripts/README.md (diagnostics)
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

# 中文：跨目录复用同一预处理实现。 English: reuse the same preprocessing implementation.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from preprocess_robotwin_flux import build_composite_batch, HDF5_CAMERAS
from robonana.encoding import encode_flux2_image_tokens
from robonana.image_pipeline import (
    VIEW_KEYS, build_robotwin_vae_input, image_contract,
)
from robonana.inference.batched_policy import BatchedRoboNanaRobotWinPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path)
    parser.add_argument("--lerobot-task", type=Path)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--verify-saved-cache", action="store_true",
                        help="Also compare the generated LeRobot episode cache against actual live encoding")
    parser.add_argument("--report", type=Path, help="Save the JSON validation evidence")
    args = parser.parse_args()
    if not args.hdf5 and not args.lerobot_task:
        parser.error("Provide --hdf5 and/or --lerobot-task")
    if args.verify_saved_cache and not args.lerobot_task:
        parser.error("--verify-saved-cache requires --lerobot-task")
    vae = AutoencoderKLFlux2.from_pretrained(
        args.checkpoint, subfolder="vae", torch_dtype=torch.float32,
        local_files_only=True).eval().requires_grad_(False).to(args.device)
    observations = []
    if args.hdf5:
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
        return cache.cpu()

    if args.hdf5:
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
        tokens = check("lerobot_original", build_robotwin_vae_input(raw), obs)
        if args.verify_saved_cache:
            # 中文：核验实际落盘产物，不仅比较两个运行时函数。
            # English: Check the persisted episode and its proof, not just two live call paths.
            from robonana.data.flux_cache import episode_cache_path
            from robonana.image_pipeline import valid_image_cache
            path = episode_cache_path(args.lerobot_task, row["episode_index"])
            assert valid_image_cache(path, row["length"]), f"Incomplete cache: {path}"
            saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            assert bool(torch.isfinite(saved).all()), f"Nonfinite cache: {path}"
            assert torch.equal(saved[:2].float(), tokens), "Saved cache differs from online encoding"
            report["saved_episode"] = dict(path=str(path), shape=list(saved.shape),
                finite=True, sampled_frames=[0,1], saved_live_equal=True, max_abs_error=0.)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
