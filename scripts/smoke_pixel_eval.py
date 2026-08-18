#!/usr/bin/env python3
"""Decode one real cached RoboTwin FLUX frame to verify pixel monitoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from robonana.training.visualization import decode_flux2_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from diffusers.models import AutoencoderKLFlux2

    device = torch.device(args.device)
    vae = AutoencoderKLFlux2.from_pretrained(
        args.checkpoint_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    vae.requires_grad_(False)
    vae.to(device)
    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=True)
    tokens = cache[args.frame_index : args.frame_index + 1].to(device)
    pixels = decode_flux2_tokens(vae, tokens)[0]
    array = pixels.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array)).save(args.output)
    print(
        f"decoded={args.output} shape={tuple(pixels.shape)} "
        f"range=({float(pixels.min()):.6f},{float(pixels.max()):.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
