#!/usr/bin/env python3
# 中文：服务入口：提供多客户端动态 batch 推理。
# English: Service entry: serve dynamically batched multi-client inference.
# 调用 / Invocation: 正式评测/采集调用；参数以 checkpoint 契约为准。 / Used by eval/collection; settings must match the checkpoint contract.
# 导航 / Guide: scripts/README.md (services)
"""Serve Stage-1 RoboNana actions with true multi-client dynamic batching."""

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

from robonana.inference.batched_policy import BatchedRoboNanaRobotWinPolicy
from robonana.inference.robotwin_policy import InferenceMode
from robonana.inference.dynamic_batch_server import DynamicBatchRobotInferenceServer
from robonana.inference.dynamic_batching import BatchMetricsPolicy
from world_action_model import apply_runtime_compat


def main() -> int:
    apply_runtime_compat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--flux-checkpoint-dir", required=True)
    parser.add_argument("--stats-path", default=str(A_STATS_PATH), help="Must be Stage-1 statistics A")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--vae-device", default="cuda:0")
    parser.add_argument("--text-encoder-device", default="cpu")
    parser.add_argument("--action-chunk", type=int, default=None, help="Must match checkpoint contract")
    parser.add_argument("--horizon", type=int, default=None, help="Must match checkpoint contract")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Must match checkpoint contract")
    parser.add_argument("--flow-shift", type=float, default=None, help="Must match checkpoint contract")
    parser.add_argument(
        "--inference-mode",
        choices=tuple(mode.value for mode in InferenceMode),
        default=InferenceMode.ACTION_Q_REJECTION.value,
    )
    parser.add_argument("--rejection-candidate-count", type=int, default=None, help="Must match checkpoint contract")
    parser.add_argument("--q-return-scale", type=float, default=None, help="Must match checkpoint contract")
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--max-batch-wait-ms", type=float, default=100.0)
    parser.add_argument("--max-clients", type=int, default=16)
    parser.add_argument("--batch-metrics-path", type=Path, default=None)
    parser.add_argument("--action-student", type=Path, default=None,
                        help="Explicit isolated student evaluation; never enabled by MAC training")
    args = parser.parse_args()
    policy_class = BatchedRoboNanaRobotWinPolicy
    extra = {}
    if args.action_student:
        from robonana.inference.student_policy import StudentRobotWinPolicy
        policy_class = StudentRobotWinPolicy
        extra['student_checkpoint'] = args.action_student
    policy = policy_class(
        **extra,
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
        inference_mode=args.inference_mode,
        rejection_candidate_count=args.rejection_candidate_count,
        q_return_scale=args.q_return_scale,
    )
    resolved = policy.load_report.model_config
    print(
        f"Loaded RoboNana checkpoint with {policy.load_report.checkpoint_parameters:,} "
        f"parameters; architecture={resolved.params.hidden_size}d/"
        f"{resolved.params.num_heads}h/{resolved.params.depth}+"
        f"{resolved.params.depth_single_blocks} blocks; source={resolved.source}",
        flush=True,
    )
    if args.batch_metrics_path is not None:
        policy = BatchMetricsPolicy(policy, args.batch_metrics_path)
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
