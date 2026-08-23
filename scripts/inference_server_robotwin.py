#!/usr/bin/env python3
"""Serve a trained RoboNana checkpoint through FACT's RoboTwin TCP protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for upstream in reversed(
    (
        REPO_ROOT / "src",
        REPO_ROOT / "third_party" / "FACT",
        REPO_ROOT / "third_party" / "flux2" / "src",
        REPO_ROOT / "third_party" / "flux2_official" / "src",
    )
):
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

import torch

from robonana.inference import RoboNanaRobotWinPolicy
from world_action_model import apply_runtime_compat
from world_action_model.sockets import RobotInferenceServer


def main() -> int:
    apply_runtime_compat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model-config",
        default=None,
        help=(
            "Optional complete training config JSON. By default config.json is discovered "
            "above the checkpoint; missing metadata is an error."
        ),
    )
    parser.add_argument("--flux-checkpoint-dir", required=True)
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--vae-device", default="cuda:1")
    parser.add_argument("--text-encoder-device", default="cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--action-chunk", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument(
        "--return-chunk-value",
        action="store_true",
        help="Run the Stage-2 world sampler and return one denormalized value per action chunk.",
    )
    parser.add_argument(
        "--return-stage2-image",
        action="store_true",
        help="Decode and return the final Stage-2 future image (requires --return-chunk-value).",
    )
    args = parser.parse_args()
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    policy = RoboNanaRobotWinPolicy(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        flux_checkpoint_dir=args.flux_checkpoint_dir,
        stats_path=args.stats_path,
        model_device=args.model_device,
        vae_device=args.vae_device,
        text_encoder_device=args.text_encoder_device,
        dtype=dtype,
        action_chunk=args.action_chunk,
        horizon=args.horizon,
        num_inference_steps=args.num_inference_steps,
        return_chunk_value=args.return_chunk_value,
        return_stage2_image=args.return_stage2_image,
    )
    resolved = policy.load_report.model_config
    print(
        f"Loaded RoboNana checkpoint with {policy.load_report.checkpoint_parameters:,} parameters; "
        f"architecture={resolved.params.hidden_size}d/{resolved.params.num_heads}h/"
        f"{resolved.params.depth}+{resolved.params.depth_single_blocks} blocks; source={resolved.source}",
        flush=True,
    )
    server = RobotInferenceServer(policy, host=args.host, port=args.port)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
