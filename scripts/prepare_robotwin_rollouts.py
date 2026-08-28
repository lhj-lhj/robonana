#!/usr/bin/env python3
"""Prepare one separate RoboTwin rollout collection for RoboNana training."""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import h5py
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    REPO_ROOT / "src",
    REPO_ROOT / "third_party" / "FACT",
    REPO_ROOT / "third_party" / "flux2" / "src",
    REPO_ROOT / "third_party" / "flux2_official" / "src",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from preprocess_robotwin_flux import (  # noqa: E402
    atomic_json_save,
    cache_images,
    cache_language,
    discover_task_dirs,
    EXPECTED_IMAGE_TOKENS,
    EXPECTED_LATENT_CHANNELS,
    HDF5_CAMERAS,
    MAX_LENGTH,
    write_manifest,
)
from robonana.data.flux_cache import (  # noqa: E402
    episode_cache_path,
    episode_language_context_path,
)
from robonana.data.robotwin_hdf5 import RoboTwinHDF5Dataset, discover_episode_records  # noqa: E402
from robonana.data.stats import write_robotwin_metadata  # noqa: E402


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-dataset-root", type=Path, default=Path("/workspace/datasets/RoboTwin/hf_dataset"))
    parser.add_argument("--stats-source", type=Path)
    parser.add_argument("--task-glob", default="*/robonana_rollout")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("all", "metadata", "language", "images"),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    initial_root = args.initial_dataset_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if _is_within(dataset_root, initial_root):
        raise ValueError(
            f"rollout dataset root must be outside the initial dataset: {dataset_root}"
        )
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    records = discover_episode_records(dataset_root, args.task_glob)
    tasks = discover_task_dirs(dataset_root, args.task_glob, max_tasks=0)
    index_path, stats_path = write_robotwin_metadata(
        dataset_root,
        task_glob=args.task_glob,
        index_path=dataset_root / "robonana_index.json",
        stats_path=dataset_root / "robonana_norm_stats.json",
    )
    if args.stats_source is not None:
        source = args.stats_source.expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["source_stats"] = str(source)
        atomic_json_save(payload, stats_path)
    if args.stage == "metadata":
        print(f"episode index: {index_path}")
        print(f"normalization stats: {stats_path}")
        return 0

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    for task_dir in tasks:
        write_manifest(task_dir, checkpoint)
    if args.stage in ("all", "language"):
        cache_language(
            tasks,
            checkpoint,
            device,
            args.overwrite,
            per_episode=True,
        )
    if args.stage in ("all", "images"):
        cache_images(
            tasks,
            checkpoint,
            device,
            args.batch_size,
            max_episodes_per_task=0,
            overwrite=args.overwrite,
            rank=0,
            world_size=1,
        )

    missing = []
    for record in records:
        latent_path = episode_cache_path(record.task_dir, record.episode_index)
        context_path = episode_language_context_path(record.task_dir, record.episode_index)
        if not latent_path.is_file():
            missing.append(f"image:{record.source}")
        else:
            latents = torch.load(latent_path, map_location="cpu", weights_only=True)
            expected_shape = (record.length, EXPECTED_IMAGE_TOKENS, EXPECTED_LATENT_CHANNELS)
            if tuple(latents.shape) != expected_shape:
                missing.append(f"image-shape:{record.source}:{tuple(latents.shape)}")
        if not context_path.is_file():
            missing.append(f"language:{record.source}")
        else:
            context = torch.load(context_path, map_location="cpu", weights_only=True)
            if tuple(context.shape) != (MAX_LENGTH, 7680):
                missing.append(f"language-shape:{record.source}:{tuple(context.shape)}")
        with h5py.File(record.source, "r") as handle:
            state_shape = tuple(handle["joint_action/vector"].shape)
            action_shape = tuple(handle["policy_action/vector"].shape)
            expected_vector_shape = (record.length, 14)
            if state_shape != expected_vector_shape or action_shape != expected_vector_shape:
                missing.append(
                    f"vector-shape:{record.source}:state={state_shape}:action={action_shape}"
                )
            if not bool(handle.attrs.get("has_final_observation", False)):
                missing.append(f"reset-pre-final-observation:{record.source}")
            if "transition_valid" not in handle:
                missing.append(f"transition-valid:{record.source}")
            else:
                transition_valid = handle["transition_valid"][:].astype(bool, copy=False)
                if transition_valid.shape != (record.length,) or bool(transition_valid[-1]):
                    missing.append(
                        f"transition-valid-shape-or-tail:{record.source}:"
                        f"shape={transition_valid.shape}:last={bool(transition_valid[-1])}"
                    )
            for camera in HDF5_CAMERAS:
                frames = handle[f"observation/{camera}/rgb"]
                if len(frames) != record.length:
                    missing.append(f"camera-length:{record.source}:{camera}:{len(frames)}")
                    continue
                if record.length:
                    with Image.open(BytesIO(bytes(frames[0]))) as image:
                        image.verify()
    if args.stage == "all" and missing:
        raise RuntimeError(f"rollout preparation left {len(missing)} missing caches: {missing[:5]}")
    sample_validation = None
    if args.stage == "all":
        dataset = RoboTwinHDF5Dataset(
            str(dataset_root),
            stats_path=str(stats_path),
            task_glob=args.task_glob,
            index_path=str(index_path),
            fixed_horizon=1,
            eval_horizons=(1,),
            q_target_mode="td_posttrain",
            episode_filter="all",
            pool_name="latest_failure",
            require_final_observation=True,
        )
        sample = dataset[0]
        sample_validation = {
            "action_shape": list(sample["action"].shape),
            "current_latent_shape": list(sample["current_latents"].shape),
            "context_shape": list(sample["context"].shape),
            "action_loss_mask": float(sample["action_loss_mask"].item()),
            "failure_episode_mask": float(sample["failure_episode_mask"].item()),
        }
        dataset.close()
    ready = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "task_glob": args.task_glob,
        "episodes": len(records),
        "frames": sum(record.length for record in records),
        "successes": sum(record.success for record in records),
        "failures": sum(not record.success for record in records),
        "index": str(index_path),
        "stats": str(stats_path),
        "stats_source": None if args.stats_source is None else str(args.stats_source.expanduser().resolve()),
        "checkpoint": str(checkpoint),
        "complete": args.stage == "all" and not missing,
        "sample_validation": sample_validation,
    }
    atomic_json_save(ready, dataset_root / "robonana_ready.json")
    print(json.dumps(ready, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
