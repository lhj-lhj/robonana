#!/usr/bin/env python3
# 中文：服务入口：提供 FACT TCP 协议的单请求推理服务。
# English: Service entry: serve single-request inference over the FACT TCP protocol.
# 调用 / Invocation: 手动启动用于集成；加载模型并监听端口，不训练。 / Manual integration service; loads models and listens on a port, never trains.
# 导航 / Guide: scripts/README.md (services)
"""Serve a trained RoboNana checkpoint through FACT's RoboTwin TCP protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
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
from robonana.normalization import A_STATS_PATH

from robonana.inference import InferenceMode, RoboNanaRobotWinPolicy
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
    parser.add_argument("--stats-path", default=str(A_STATS_PATH), help="Must be Stage-1 statistics A")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--vae-device", default="cuda:1")
    parser.add_argument("--text-encoder-device", default="cpu")
    parser.add_argument("--action-chunk", type=int, default=None, help="Must match checkpoint contract")
    parser.add_argument("--horizon", type=int, default=None, help="Must match checkpoint contract")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Must match checkpoint contract")
    parser.add_argument("--flow-shift", type=float, default=None, help="Must match checkpoint contract")
    parser.add_argument("--discount", type=float, default=None, help="Must match checkpoint contract")
    parser.add_argument("--reward-non-goal", type=float, default=None, help="Must match checkpoint contract")
    parser.add_argument("--success-threshold", type=float, default=None, help="Must match checkpoint contract")
    parser.add_argument("--rejection-candidate-count", type=int, default=None, help="Must match checkpoint contract")
    parser.add_argument("--q-return-scale", type=float, default=None, help="Must match checkpoint contract")
    parser.add_argument(
        "--inference-mode",
        choices=tuple(mode.value for mode in InferenceMode),
        default=InferenceMode.ACTION_Q_REJECTION.value,
        help="Sample 48-step action candidates and select argmax Q.",
    )
    parser.add_argument(
        "--vae-decode-batch-size",
        type=int,
        default=4,
        help="Number of generated horizon latents decoded by the VAE at once.",
    )
    args = parser.parse_args()
    policy = RoboNanaRobotWinPolicy(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        flux_checkpoint_dir=args.flux_checkpoint_dir,
        stats_path=args.stats_path,
        model_device=args.model_device,
        vae_device=args.vae_device,
        text_encoder_device=args.text_encoder_device,
        dtype=torch.bfloat16,
        action_chunk=args.action_chunk,
        horizon=args.horizon,
        num_inference_steps=args.num_inference_steps,
        flow_shift=args.flow_shift,
        discount=args.discount,
        reward_non_goal=args.reward_non_goal,
        success_threshold=args.success_threshold,
        rejection_candidate_count=args.rejection_candidate_count,
        q_return_scale=args.q_return_scale,
        inference_mode=args.inference_mode,
        vae_decode_batch_size=args.vae_decode_batch_size,
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
