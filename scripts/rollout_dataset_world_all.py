#!/usr/bin/env python3
"""Roll RoboNana forward from first frames of training-set trajectories."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from robonana.data.robotwin_lerobot import (
    RoboTwinLeRobotDataset,
    lerobot_episode_instruction,
)
from robonana.inference.robotwin_policy import InferenceMode, RoboNanaRobotWinPolicy
from robonana.inference.rollout_artifacts import (
    annotate_rollout_frame,
    decoded_frame_to_uint8,
    split_robotwin_composite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--flux-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--vae-device", default="cuda:0")
    parser.add_argument("--trajectory-count", type=int, default=10)
    parser.add_argument("--rollout-rounds", type=int, default=5)
    parser.add_argument("--split", default="Clean")
    parser.add_argument("--selection-seed", type=int, default=20260827)
    parser.add_argument("--sampling-seed", type=int, default=2026082700)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--stage2-image-horizon-batch-size", type=int, default=2)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_training_records(dataset: RoboTwinLeRobotDataset, args: argparse.Namespace):
    grouped = defaultdict(list)
    for record in dataset.records:
        if record.task_dir.parent.name.casefold() == args.split.casefold():
            grouped[record.task_name].append(record)
    task_names = sorted(grouped)
    if args.trajectory_count > len(task_names):
        raise ValueError(
            f"requested {args.trajectory_count} distinct tasks from {args.split}, "
            f"but only {len(task_names)} are available"
        )
    rng = random.Random(args.selection_seed)
    rng.shuffle(task_names)
    selected = []
    for task_name in task_names[: args.trajectory_count]:
        candidates = sorted(grouped[task_name], key=lambda record: record.episode_index)
        selected.append(rng.choice(candidates))
    return selected


def first_observation(dataset: RoboTwinLeRobotDataset, record) -> dict[str, Any]:
    states, _ = dataset._episode_state_action(record)
    images = dataset._future_dino_images(record, 0)
    return {
        "observation.state": torch.from_numpy(states[0].copy()),
        **images,
        "instruction": lerobot_episode_instruction(record.task_dir, record.episode_index),
    }


def validate_response(response: dict[str, Any], *, action_chunk: int = 48) -> None:
    expected = {
        "action": (action_chunk, 14),
        "future_states": (action_chunk, 14),
        "values": (action_chunk,),
        "future_latents": (action_chunk, 288, 128),
        "images": (1, 3, action_chunk, 192, 384),
    }
    for key, shape in expected.items():
        value = torch.as_tensor(response[key])
        if tuple(value.shape) != shape:
            raise RuntimeError(f"{key} has shape {tuple(value.shape)}, expected {shape}")
        if not torch.isfinite(value.float()).all():
            raise RuntimeError(f"{key} contains non-finite values")


def run_trajectory(
    *,
    policy: RoboNanaRobotWinPolicy,
    dataset: RoboTwinLeRobotDataset,
    record,
    trajectory_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trajectory_dir = args.output_dir / f"trajectory_{trajectory_index:02d}"
    complete_path = trajectory_dir / "complete.json"
    if complete_path.is_file():
        print(f"[trajectory {trajectory_index:02d}] already complete", flush=True)
        return json.loads(complete_path.read_text(encoding="utf-8"))
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    observation = first_observation(dataset, record)
    instruction = str(observation["instruction"])
    rounds = []
    video_path = trajectory_dir / "rollout_5x48_value_overlay.mp4"
    writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8)
    started = time.perf_counter()
    try:
        for rollout_index in range(args.rollout_rounds):
            round_seed = args.sampling_seed + trajectory_index * 100_000 + rollout_index * 1_000
            policy.inference_mode = InferenceMode.ACTION
            action_observation = {**observation, "sampling_seed": round_seed}
            action_response = policy.inference(action_observation)
            action = torch.as_tensor(action_response["action"]).float().cpu()

            policy.inference_mode = InferenceMode.WORLD_ALL
            world_observation = {
                **observation,
                "action_chunk": action,
                "sampling_seed": round_seed + 1,
            }
            world_response = policy.inference(world_observation)
            validate_response(world_response)
            values = torch.as_tensor(world_response["values"]).float().cpu()
            states = torch.as_tensor(world_response["future_states"]).float().cpu()
            latents = torch.as_tensor(world_response["future_latents"]).to(dtype=torch.bfloat16)
            decoded = torch.as_tensor(world_response["images"])[0].cpu()

            round_dir = trajectory_dir / f"round_{rollout_index:02d}"
            frames_dir = round_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for horizon in range(1, 49):
                raw_frame = decoded[:, horizon - 1]
                frame_uint8 = decoded_frame_to_uint8(raw_frame)
                annotated = annotate_rollout_frame(
                    frame_uint8,
                    trajectory_index=trajectory_index,
                    rollout_index=rollout_index,
                    rollout_count=args.rollout_rounds,
                    horizon=horizon,
                    value=float(values[horizon - 1].item()),
                )
                raw_image = Image.fromarray(
                    frame_uint8.permute(1, 2, 0).contiguous().numpy(),
                    mode="RGB",
                )
                raw_image.save(frames_dir / f"frame_{horizon:03d}.png")
                writer.append_data(np.asarray(annotated))

            torch.save(
                {
                    "action": action,
                    "future_states": states,
                    "values": values,
                    "future_latents": latents,
                },
                round_dir / "predictions.pt",
            )
            rounds.append(
                {
                    "round": rollout_index,
                    "action_sampling_seed": round_seed,
                    "world_sampling_seed": round_seed + 1,
                    "values": values.tolist(),
                    "action": action.tolist(),
                    "final_state": states[-1].tolist(),
                    "timing_ms": {
                        "action": action_response.get("_policy_timing_ms", {}),
                        "world_all": world_response.get("_policy_timing_ms", {}),
                    },
                }
            )

            final_frame_uint8 = decoded_frame_to_uint8(decoded[:, -1])
            observation = {
                "observation.state": states[-1],
                **split_robotwin_composite(final_frame_uint8),
                "instruction": instruction,
            }
            print(
                f"[trajectory {trajectory_index:02d}] round={rollout_index + 1}/"
                f"{args.rollout_rounds} final_value={float(values[-1]):.5f}",
                flush=True,
            )
    finally:
        writer.close()

    payload = {
        "trajectory_index": trajectory_index,
        "task_name": record.task_name,
        "split": record.task_dir.parent.name,
        "episode_index": record.episode_index,
        "episode_length": record.length,
        "instruction": instruction,
        "rollout_rounds": args.rollout_rounds,
        "frames_per_round": 48,
        "video": video_path.name,
        "elapsed_seconds": time.perf_counter() - started,
        "rounds": rounds,
    }
    write_json(complete_path, payload)
    return payload


def main() -> None:
    args = parse_args()
    if args.trajectory_count <= 0 or args.rollout_rounds <= 0:
        raise ValueError("trajectory-count and rollout-rounds must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must lie in [0, num-shards)")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = RoboTwinLeRobotDataset(
        data_path=str(args.dataset_root),
        stats_path=str(args.dataset_root / "robonana_norm_stats.json"),
        index_path=str(args.dataset_root / "robonana_index.json"),
        task_globs=("Clean/*", "Randomized/*"),
        action_chunk=48,
        action_dim=14,
        max_horizon=48,
        dino_online=False,
    )
    dataset.open()
    selected = select_training_records(dataset, args)
    assigned = [
        (index, record)
        for index, record in enumerate(selected)
        if index % args.num_shards == args.shard_id
    ]
    write_json(
        args.output_dir / f"selection_shard_{args.shard_id:02d}.json",
        {
            "selection_seed": args.selection_seed,
            "sampling_seed": args.sampling_seed,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "assigned": [
                {
                    "trajectory_index": index,
                    "task_name": record.task_name,
                    "split": record.task_dir.parent.name,
                    "episode_index": record.episode_index,
                }
                for index, record in assigned
            ],
        },
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(args.model_device))
    policy = RoboNanaRobotWinPolicy(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        flux_checkpoint_dir=args.flux_checkpoint_dir,
        stats_path=args.dataset_root / "robonana_norm_stats.json",
        model_device=args.model_device,
        vae_device=args.vae_device,
        text_encoder_device="cpu",
        dtype=torch.bfloat16,
        action_chunk=48,
        num_inference_steps=args.num_inference_steps,
        inference_mode=InferenceMode.WORLD_ALL,
        stage2_image_horizon_batch_size=args.stage2_image_horizon_batch_size,
        vae_decode_batch_size=args.vae_decode_batch_size,
    )
    try:
        for trajectory_index, record in assigned:
            run_trajectory(
                policy=policy,
                dataset=dataset,
                record=record,
                trajectory_index=trajectory_index,
                args=args,
            )
        peak_bytes = (
            torch.cuda.max_memory_reserved(torch.device(args.model_device))
            if torch.cuda.is_available()
            else 0
        )
        write_json(
            args.output_dir / f"complete_shard_{args.shard_id:02d}.json",
            {
                "status": "complete",
                "shard_id": args.shard_id,
                "trajectory_indices": [index for index, _ in assigned],
                "peak_cuda_reserved_bytes": int(peak_bytes),
            },
        )
    except Exception as error:
        write_json(
            args.output_dir / f"failed_shard_{args.shard_id:02d}.json",
            {"status": "failed", "shard_id": args.shard_id, "error": repr(error)},
        )
        raise
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
