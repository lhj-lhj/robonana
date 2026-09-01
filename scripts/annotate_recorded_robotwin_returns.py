#!/usr/bin/env python3
"""Annotate recorded RoboTwin frames with packed-horizon reward/Q predictions."""

from __future__ import annotations

import argparse
import json
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from robonana.data.robotwin_hdf5 import EpisodeRecord, RoboTwinHDF5Dataset
from robonana.data.robotwin_lerobot import (
    RoboTwinLeRobotDataset,
    _video_path,
    lerobot_episode_instruction,
)
from robonana.inference.robotwin_policy import InferenceMode, RoboNanaRobotWinPolicy
from robonana.inference.rollout_artifacts import annotate_recorded_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--flux-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-format", choices=("lerobot", "hdf5"), required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--group-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--action-chunk", type=int, default=48)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--sampling-seed", type=int, default=2026090200)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--vae-device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    return parser.parse_args()


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    if not result:
        raise ValueError("group name does not contain a filename-safe character")
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _metadata_path(record: EpisodeRecord) -> Path:
    return record.task_dir / "metadata" / f"episode{record.episode_index}.json"


def _record_metadata(record: EpisodeRecord, dataset_format: str) -> dict[str, Any]:
    if dataset_format == "lerobot":
        return {
            "instruction": lerobot_episode_instruction(record.task_dir, record.episode_index),
            "seed": None,
            "success": True,
        }
    path = _metadata_path(record)
    if not path.is_file():
        raise FileNotFoundError(f"missing rollout metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    instruction = str(payload.get("instruction", "")).strip()
    if not instruction:
        raise ValueError(f"rollout instruction is empty: {path}")
    return {
        **payload,
        "instruction": instruction,
        "seed": None if payload.get("seed") is None else int(payload["seed"]),
        "success": bool(payload.get("success", record.success)),
    }


def _build_dataset(args: argparse.Namespace):
    common = dict(
        data_path=str(args.dataset_root),
        stats_path=str(args.stats_path),
        action_chunk=args.action_chunk,
        action_dim=14,
        max_horizon=args.action_chunk,
        dino_online=False,
        hdf5_cache_size=1,
        latent_cache_size=1,
        language_cache_size=1,
    )
    if args.dataset_format == "lerobot":
        dataset = RoboTwinLeRobotDataset(
            **common,
            task_globs=(f"Clean/{args.task_name}",),
            index_path=str(args.dataset_root / "robonana_index.json"),
        )
    else:
        dataset = RoboTwinHDF5Dataset(
            **common,
            task_glob=f"{args.task_name}/robonana_rollout",
            index_path=str(args.dataset_root / "robonana_index.json"),
            q_target_mode="td_posttrain",
            episode_filter="all",
            pool_name="latest_failure",
        )
    dataset.open()
    records = [record for record in dataset.records if record.task_name == args.task_name]
    if args.dataset_format == "lerobot":
        records = [
            record
            for record in records
            if record.task_dir.parent.name.casefold() == "clean"
        ]
    records = sorted(records, key=lambda item: item.episode_index)
    if len(records) != args.expected_episodes:
        dataset.close()
        raise RuntimeError(
            f"{args.group_name} contains {len(records)} episodes, "
            f"expected exactly {args.expected_episodes}"
        )
    return dataset, records


def _instruction_observation(
    dataset,
    record: EpisodeRecord,
    *,
    frame_index: int,
    state: np.ndarray,
    instruction: str,
) -> dict[str, Any]:
    images = dataset._future_dino_images(record, frame_index)
    return {
        "observation.state": torch.from_numpy(state[frame_index].copy()),
        **images,
        "instruction": instruction,
    }


def _padded_action_chunk(actions: np.ndarray, start: int, action_chunk: int) -> torch.Tensor:
    if actions.ndim != 2 or not len(actions):
        raise ValueError("episode actions must be a non-empty [frames, action_dim] array")
    indices = np.clip(np.arange(start, start + action_chunk), 0, len(actions) - 1)
    return torch.from_numpy(actions[indices].copy())


def _score_episode(
    *,
    policy: RoboNanaRobotWinPolicy,
    dataset,
    record: EpisodeRecord,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    partial_path: Path,
) -> dict[str, Any]:
    states, actions = dataset._episode_state_action(record)
    if len(states) != record.length or len(actions) != record.length:
        raise RuntimeError(f"state/action length mismatch for {record.source}")
    rewards = np.full(record.length, np.nan, dtype=np.float32)
    qs = np.full(record.length, np.nan, dtype=np.float32)
    chunk_ids = np.full(record.length, -1, dtype=np.int32)
    horizons = np.zeros(record.length, dtype=np.int16)
    chunks: list[dict[str, Any]] = []
    started = time.perf_counter()
    for chunk_index, start in enumerate(range(0, max(record.length - 1, 0), args.action_chunk)):
        observation = _instruction_observation(
            dataset,
            record,
            frame_index=start,
            state=states,
            instruction=metadata["instruction"],
        )
        observation.update(
            action_chunk=_padded_action_chunk(actions, start, args.action_chunk),
            include_image=False,
            sampling_seed=(
                args.sampling_seed
                + record.episode_index * 100_000
                + chunk_index * 1_000
            ),
        )
        response = policy.inference(observation)
        response_horizons = torch.as_tensor(response["horizons"]).tolist()
        if response_horizons != list(range(1, args.action_chunk + 1)):
            raise RuntimeError(f"unexpected horizons: {response_horizons}")
        chunk_rewards = torch.as_tensor(response["rewards"]).float().numpy()
        chunk_qs = torch.as_tensor(response["qs"]).float().numpy()
        for horizon in range(1, args.action_chunk + 1):
            frame_index = start + horizon
            if frame_index >= record.length:
                break
            rewards[frame_index] = chunk_rewards[horizon - 1]
            qs[frame_index] = chunk_qs[horizon - 1]
            chunk_ids[frame_index] = chunk_index
            horizons[frame_index] = horizon
        chunks.append(
            {
                "chunk_index": chunk_index,
                "start_frame": start,
                "sampling_seed": observation["sampling_seed"],
                "valid_horizons": min(args.action_chunk, record.length - 1 - start),
            }
        )
        _write_json(
            partial_path,
            {
                "status": "scoring",
                "episode_index": record.episode_index,
                "completed_chunks": len(chunks),
                "total_chunks": (record.length - 2) // args.action_chunk + 1,
            },
        )
        print(
            f"[{args.group_name} episode={record.episode_index:03d}] "
            f"chunk={chunk_index + 1:03d} start={start:04d}",
            flush=True,
        )
    return {
        "rewards": rewards,
        "qs": qs,
        "chunk_ids": chunk_ids,
        "horizons": horizons,
        "chunks": chunks,
        "score_seconds": time.perf_counter() - started,
    }


def _iter_source_frames(
    dataset,
    record: EpisodeRecord,
    dataset_format: str,
) -> Iterable[torch.Tensor]:
    if dataset_format == "lerobot":
        reader = imageio.get_reader(
            _video_path(record.task_dir, record.episode_index, "observation.images.cam_high")
        )
        try:
            for array in reader:
                yield torch.from_numpy(np.asarray(array, dtype=np.uint8).copy()).permute(2, 0, 1)
        finally:
            reader.close()
        return
    handle: h5py.File = dataset._handle(record.source)
    frames = handle["observation/head_camera/rgb"]
    for encoded in frames:
        with Image.open(BytesIO(bytes(encoded))) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        yield torch.from_numpy(array).permute(2, 0, 1)


def _render_episode(
    *,
    dataset,
    record: EpisodeRecord,
    metadata: dict[str, Any],
    telemetry: dict[str, Any],
    args: argparse.Namespace,
    video_path: Path,
) -> int:
    writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8)
    rendered = 0
    try:
        for frame_index, frame in enumerate(
            _iter_source_frames(dataset, record, args.dataset_format)
        ):
            if frame_index >= record.length:
                break
            reward = None if frame_index == 0 else telemetry["rewards"][frame_index]
            q = None if frame_index == 0 else telemetry["qs"][frame_index]
            annotated = annotate_recorded_frame(
                frame,
                group=args.group_name,
                episode_index=record.episode_index,
                frame_index=frame_index,
                action_chunk=args.action_chunk,
                reward=reward,
                q=q,
            )
            writer.append_data(np.asarray(annotated))
            rendered += 1
    finally:
        writer.close()
    if rendered != record.length:
        video_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"rendered {rendered} frames for {record.source}, expected {record.length}"
        )
    return rendered


def _episode_filename(group_name: str, record: EpisodeRecord, metadata: dict[str, Any]) -> str:
    seed = "na" if metadata.get("seed") is None else str(int(metadata["seed"]))
    success = int(bool(metadata.get("success", record.success)))
    return (
        f"{_safe_name(group_name)}__episode_{record.episode_index:03d}"
        f"__seed_{seed}__success_{success}.mp4"
    )


def main() -> None:
    args = parse_args()
    if args.expected_episodes <= 0 or args.action_chunk <= 0:
        raise ValueError("expected-episodes and action-chunk must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must lie in [0, num-shards)")
    args.output_dir = args.output_dir.expanduser().resolve()
    video_dir = args.output_dir / "videos"
    telemetry_dir = args.output_dir / "telemetry"
    status_dir = args.output_dir / "status"
    for path in (video_dir, telemetry_dir, status_dir):
        path.mkdir(parents=True, exist_ok=True)

    dataset, records = _build_dataset(args)
    assigned = [record for index, record in enumerate(records) if index % args.num_shards == args.shard_id]
    policy = RoboNanaRobotWinPolicy(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        flux_checkpoint_dir=args.flux_checkpoint_dir,
        stats_path=args.stats_path,
        model_device=args.model_device,
        vae_device=args.vae_device,
        text_encoder_device="cpu",
        dtype=torch.bfloat16,
        action_chunk=args.action_chunk,
        num_inference_steps=args.num_inference_steps,
        inference_mode=InferenceMode.WORLD_ALL,
    )
    rows = []
    try:
        for record in assigned:
            metadata = _record_metadata(record, args.dataset_format)
            stem = Path(_episode_filename(args.group_name, record, metadata)).stem
            video_path = video_dir / f"{stem}.mp4"
            telemetry_path = telemetry_dir / f"{stem}.json"
            complete_path = status_dir / f"{stem}.complete.json"
            partial_path = status_dir / f"{stem}.partial.json"
            if complete_path.is_file() and video_path.is_file() and telemetry_path.is_file():
                rows.append(json.loads(complete_path.read_text(encoding="utf-8")))
                print(f"[{args.group_name} episode={record.episode_index:03d}] complete; skip", flush=True)
                continue
            telemetry = _score_episode(
                policy=policy,
                dataset=dataset,
                record=record,
                metadata=metadata,
                args=args,
                partial_path=partial_path,
            )
            rendered = _render_episode(
                dataset=dataset,
                record=record,
                metadata=metadata,
                telemetry=telemetry,
                args=args,
                video_path=video_path,
            )
            valid = np.isfinite(telemetry["qs"])
            telemetry_payload = {
                "group": args.group_name,
                "episode_index": record.episode_index,
                "seed": metadata.get("seed"),
                "success": bool(metadata.get("success", record.success)),
                "instruction": metadata["instruction"],
                "source": str(record.source),
                "checkpoint": str(args.checkpoint),
                "action_chunk": args.action_chunk,
                "frame_index": list(range(record.length)),
                "chunk_index": telemetry["chunk_ids"].tolist(),
                "horizon": telemetry["horizons"].tolist(),
                "reward_h": [None if not np.isfinite(v) else float(v) for v in telemetry["rewards"]],
                "q_h": [None if not np.isfinite(v) else float(v) for v in telemetry["qs"]],
                "chunks": telemetry["chunks"],
            }
            _write_json(telemetry_path, telemetry_payload)
            row = {
                "group": args.group_name,
                "episode_index": record.episode_index,
                "seed": metadata.get("seed"),
                "success": bool(metadata.get("success", record.success)),
                "frames": rendered,
                "chunks": len(telemetry["chunks"]),
                "reward_mean": float(telemetry["rewards"][valid].mean()),
                "q_mean": float(telemetry["qs"][valid].mean()),
                "video": str(video_path.relative_to(args.output_dir)),
                "telemetry": str(telemetry_path.relative_to(args.output_dir)),
                "score_seconds": telemetry["score_seconds"],
            }
            _write_json(complete_path, row)
            partial_path.unlink(missing_ok=True)
            rows.append(row)
    finally:
        dataset.close()
    manifest_path = args.output_dir / f"manifest_{_safe_name(args.group_name)}_shard_{args.shard_id:02d}.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(
        args.output_dir / f"complete_{_safe_name(args.group_name)}_shard_{args.shard_id:02d}.json",
        {
            "status": "complete",
            "group": args.group_name,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "episodes": len(rows),
        },
    )


if __name__ == "__main__":
    main()
