#!/usr/bin/env python3
"""Serve Stage-1 RoboNana actions with true multi-client dynamic batching."""

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

from robonana.inference.batched_policy import BatchedRoboNanaRobotWinPolicy
from robonana.inference.dynamic_batch_server import DynamicBatchRobotInferenceServer
from world_action_model import apply_runtime_compat


def main() -> int:
    apply_runtime_compat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--flux-checkpoint-dir", required=True)
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--vae-device", default="cuda:0")
    parser.add_argument("--text-encoder-device", default="cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--action-chunk", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--max-batch-wait-ms", type=float, default=100.0)
    parser.add_argument("--max-clients", type=int, default=16)
    args = parser.parse_args()
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    policy = BatchedRoboNanaRobotWinPolicy(
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
    )
    resolved = policy.load_report.model_config
    print(
        f"Loaded RoboNana checkpoint with {policy.load_report.checkpoint_parameters:,} "
        f"parameters; architecture={resolved.params.hidden_size}d/"
        f"{resolved.params.num_heads}h/{resolved.params.depth}+"
        f"{resolved.params.depth_single_blocks} blocks; source={resolved.source}",
        flush=True,
    )
    server = DynamicBatchRobotInferenceServer(
        policy,
        host=args.host,
        port=args.port,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_batch_wait_ms,
        max_clients=args.max_clients,
    )
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
