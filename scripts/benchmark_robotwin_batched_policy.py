#!/usr/bin/env python3
"""Benchmark true Stage-1 batching on real RoboTwin observations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


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

from robonana.data.robotwin_lerobot import (  # noqa: E402
    RoboTwinLeRobotDataset,
    lerobot_episode_instruction,
)
from robonana.inference.batched_policy import BatchedRoboNanaRobotWinPolicy  # noqa: E402
from world_action_model import apply_runtime_compat  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--flux-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--vae-device", default="cuda:0")
    parser.add_argument("--text-encoder-device", default="cpu")
    return parser.parse_args()


def first_observation(dataset: RoboTwinLeRobotDataset, record) -> dict[str, object]:
    states, _ = dataset._episode_state_action(record)
    images = dataset._future_dino_images(record, 0)
    return {
        "observation.state": torch.from_numpy(states[0].copy()),
        **images,
        "instruction": lerobot_episode_instruction(record.task_dir, record.episode_index),
        "sampling_seed": 2026082900 + int(record.episode_index),
    }


def main() -> int:
    apply_runtime_compat()
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    dataset = RoboTwinLeRobotDataset(
        str(args.dataset_root.expanduser().resolve()),
        stats_path=str(args.stats_path.expanduser().resolve()),
        index_path=(
            None if args.index_path is None else str(args.index_path.expanduser().resolve())
        ),
        task_globs=("Clean/*", "Randomized/*"),
        action_chunk=48,
        action_dim=14,
        max_horizon=48,
    )
    dataset.open()
    if len(dataset.records) < args.batch_size:
        raise RuntimeError(
            f"dataset contains {len(dataset.records)} episodes, need {args.batch_size}"
        )
    observations = [
        first_observation(dataset, record) for record in dataset.records[: args.batch_size]
    ]
    dataset.close()

    policy = BatchedRoboNanaRobotWinPolicy(
        checkpoint=args.checkpoint.expanduser().resolve(),
        model_config=args.model_config.expanduser().resolve(),
        flux_checkpoint_dir=args.flux_checkpoint_dir.expanduser().resolve(),
        stats_path=args.stats_path.expanduser().resolve(),
        model_device=args.model_device,
        vae_device=args.vae_device,
        text_encoder_device=args.text_encoder_device,
        dtype=torch.bfloat16,
        action_chunk=48,
        horizon=24,
        num_inference_steps=args.num_inference_steps,
    )

    cuda_device = torch.device(args.model_device)
    if cuda_device.type == "cuda":
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        baseline_bytes = torch.cuda.memory_allocated(cuda_device)
    else:
        baseline_bytes = 0

    started = time.perf_counter()
    responses = policy.inference_batch(observations)
    if cuda_device.type == "cuda":
        torch.cuda.synchronize(cuda_device)
        peak_bytes = torch.cuda.max_memory_allocated(cuda_device)
    else:
        peak_bytes = 0
    elapsed = time.perf_counter() - started

    shapes = []
    for response in responses:
        action = torch.as_tensor(response["action"])
        if tuple(action.shape) != (48, 14) or not torch.isfinite(action).all():
            raise RuntimeError(f"invalid action output: shape={tuple(action.shape)}")
        shapes.append(list(action.shape))
    report = {
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "elapsed_seconds": elapsed,
        "samples_per_second": args.batch_size / elapsed,
        "baseline_allocated_gib": baseline_bytes / 2**30,
        "peak_allocated_gib": peak_bytes / 2**30,
        "incremental_peak_gib": max(0, peak_bytes - baseline_bytes) / 2**30,
        "action_shapes": shapes,
        "policy_timing_ms": responses[0].get("_policy_timing_ms", {}),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
