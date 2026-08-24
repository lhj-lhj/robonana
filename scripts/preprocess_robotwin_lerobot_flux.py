#!/usr/bin/env python3
"""Cache Qwen3 contexts and per-frame FLUX.2 tokens for FACT RoboTwin-v2."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for upstream in (
    REPO_ROOT / "third_party" / "FACT",
    REPO_ROOT / "third_party" / "flux2" / "src",
    REPO_ROOT / "third_party" / "flux2_official" / "src",
):
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

from diffusers.models import AutoencoderKLFlux2
from flux2.text_encoder import MAX_LENGTH
from robonana.data.flux_cache import (
    CACHE_SCHEMA_VERSION,
    DINO_FEATURE_DIM,
    DINO_TOKEN_COUNT,
    episode_cache_path,
    episode_dino_cache_path,
    episode_language_context_path,
)
from robonana.data.robotwin_lerobot import (
    DEFAULT_TASK_GLOBS,
    discover_lerobot_episode_records,
)
from robonana.encoding import (
    DinoV3FeatureEncoder,
    LocalQwen3Embedder,
    encode_flux2_image_tokens,
)
from world_action_model.image_layouts import ROBOTWIN_VIEW_KEYS


MAIN_VIEW_SIZE = (256, 192)
EXPECTED_IMAGE_SHAPE = (288, 128)
EXPECTED_DINO_SHAPE = (DINO_TOKEN_COUNT, DINO_FEATURE_DIM)


_fact_preprocess_path = REPO_ROOT / "third_party" / "FACT" / "scripts" / "compute_vae_latents.py"
_fact_preprocess_spec = importlib.util.spec_from_file_location(
    "fact_compute_vae_latents",
    _fact_preprocess_path,
)
if _fact_preprocess_spec is None or _fact_preprocess_spec.loader is None:
    raise ImportError(f"Cannot load FACT preprocessing helpers from {_fact_preprocess_path}")
_fact_preprocess = importlib.util.module_from_spec(_fact_preprocess_spec)
_fact_preprocess_spec.loader.exec_module(_fact_preprocess)
_assert_frame_index_contiguous = _fact_preprocess._assert_frame_index_contiguous
_build_composite = _fact_preprocess._build_composite
_decode_view_frames = _fact_preprocess._decode_view_frames


def atomic_torch_save(value: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def episode_instructions(task_dir: Path) -> dict[int, str]:
    output = {}
    with (task_dir / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks = row.get("tasks", [])
            if isinstance(tasks, str):
                prompt = tasks.strip()
            else:
                prompt = next(
                    (value.strip() for value in tasks if isinstance(value, str) and value.strip()),
                    "",
                )
            output[int(row["episode_index"])] = prompt or task_dir.name.replace("_", " ")
    return output


def write_manifests(tasks: list[Path], checkpoint: Path, dino_model: str) -> None:
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_format": "LeRobot v2 parquet + three-view MP4",
        "layout": {
            "main_view": [256, 192],
            "side_views": [128, 96],
            "canvas": [384, 192],
        },
        "language": {
            "encoder": "Qwen3-4B",
            "shape": [MAX_LENGTH, 7680],
            "dtype": "bfloat16",
            "scope": "per_episode",
        },
        "image": {
            "encoder": "FLUX.2 AutoencoderKLFlux2",
            "shape_per_frame": list(EXPECTED_IMAGE_SHAPE),
            "dtype": "bfloat16",
        },
        "dino": {
            "encoder": dino_model,
            "input_per_view": [224, 224],
            "native_shape_per_view": [196, 768],
            "pixel_unshuffle": 2,
            "shape_per_frame": list(EXPECTED_DINO_SHAPE),
            "dtype": "bfloat16",
        },
        "checkpoint": str(checkpoint),
    }
    for task in tasks:
        atomic_json_save(manifest, task / "flux_cache" / "_manifest.json")


@torch.inference_mode()
def cache_language(
    records,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> None:
    pending: list[tuple[Path, int, str, Path]] = []
    instructions_by_task = {
        task: episode_instructions(task)
        for task in sorted({record.task_dir for record in records})
    }
    for record in records:
        prompt = instructions_by_task[record.task_dir][record.episode_index]
        output = episode_language_context_path(record.task_dir, record.episode_index)
        if overwrite or not output.is_file():
            pending.append((record.task_dir, record.episode_index, prompt, output))
    print(f"[language] pending={len(pending)} device={device} batch_size={batch_size}", flush=True)
    if not pending:
        return
    embedder = LocalQwen3Embedder(checkpoint, device)
    for start in range(0, len(pending), batch_size):
        rows = pending[start : start + batch_size]
        contexts = embedder([row[2] for row in rows]).detach().to(device="cpu", dtype=torch.bfloat16)
        if tuple(contexts.shape[1:]) != (MAX_LENGTH, 7680):
            raise RuntimeError(f"Unexpected Qwen3 context batch shape: {tuple(contexts.shape)}")
        for context, (task, episode_index, prompt, output) in zip(contexts, rows, strict=True):
            atomic_torch_save(context.contiguous(), output)
            atomic_json_save(
                {"prompt": prompt, "shape": list(context.shape), "dtype": "bfloat16"},
                output.with_suffix(".json"),
            )
        completed = min(start + len(rows), len(pending))
        if completed % 100 == 0 or completed == len(pending):
            print(f"[language] {completed}/{len(pending)}", flush=True)
    del embedder
    gc.collect()
    torch.cuda.empty_cache()


def valid_image_cache(path: Path, length: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        return False
    return tuple(value.shape) == (length, *EXPECTED_IMAGE_SHAPE) and value.dtype == torch.bfloat16


def decode_views(task_dir: Path, episode_index: int, length: int, pyav_threads: int):
    _assert_frame_index_contiguous(task_dir, episode_index, length)
    with ThreadPoolExecutor(max_workers=len(ROBOTWIN_VIEW_KEYS)) as pool:
        futures = {
            key: pool.submit(
                _decode_view_frames,
                task_dir,
                key,
                episode_index,
                pyav_threads,
                None,
            )
            for key in ROBOTWIN_VIEW_KEYS
        }
        views = {key: futures[key].result() for key in ROBOTWIN_VIEW_KEYS}
    for key, frames in views.items():
        if frames.shape[0] < length:
            raise RuntimeError(f"Video shorter than metadata for {task_dir} episode={episode_index} {key}")
        views[key] = frames[:length]
    return views


def decode_composite(task_dir: Path, episode_index: int, length: int, pyav_threads: int) -> torch.Tensor:
    views = decode_views(task_dir, episode_index, length, pyav_threads)
    return _build_composite(views, MAIN_VIEW_SIZE, list(ROBOTWIN_VIEW_KEYS))


@torch.inference_mode()
def cache_images(
    records,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    pyav_threads: int,
    overwrite: bool,
    rank: int,
    world_size: int,
) -> None:
    assigned = records[rank::world_size]
    pending = [
        record
        for record in assigned
        if overwrite or not valid_image_cache(episode_cache_path(record.task_dir, record.episode_index), record.length)
    ]
    print(f"[images rank={rank}] assigned={len(assigned)} pending={len(pending)} device={device}", flush=True)
    if not pending:
        return
    vae = AutoencoderKLFlux2.from_pretrained(
        checkpoint,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False).to(device)
    started = time.monotonic()
    for position, record in enumerate(pending, start=1):
        composite = decode_composite(
            record.task_dir,
            record.episode_index,
            record.length,
            pyav_threads,
        )
        parts = []
        for start in range(0, record.length, batch_size):
            images = composite[start : start + batch_size].to(device=device, dtype=torch.float32)
            parts.append(
                encode_flux2_image_tokens(vae, images)
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
            )
        latents = torch.cat(parts, dim=0)
        expected = (record.length, *EXPECTED_IMAGE_SHAPE)
        if tuple(latents.shape) != expected:
            raise RuntimeError(f"Unexpected cache shape {tuple(latents.shape)} != {expected}: {record.source}")
        atomic_torch_save(latents, episode_cache_path(record.task_dir, record.episode_index))
        if position % 10 == 0 or position == len(pending):
            elapsed = time.monotonic() - started
            rate = position / max(elapsed, 1e-6)
            eta = (len(pending) - position) / max(rate, 1e-6)
            print(
                f"[images rank={rank}] {position}/{len(pending)} rate={rate:.3f}ep/s eta={eta:.0f}s "
                f"task={record.task_dir.parent.name}/{record.task_dir.name} episode={record.episode_index}",
                flush=True,
            )


def valid_dino_cache(path: Path, length: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        return False
    return tuple(value.shape) == (length, *EXPECTED_DINO_SHAPE) and value.dtype == torch.bfloat16


def check_dino_cache_capacity(
    records,
    dataset_root: Path,
    *,
    overwrite: bool,
    minimum_free_gib: float,
    allow_low_disk: bool,
) -> None:
    pending = [
        record
        for record in records
        if overwrite
        or not episode_dino_cache_path(record.task_dir, record.episode_index).is_file()
    ]
    required = sum(record.length for record in pending) * DINO_TOKEN_COUNT * DINO_FEATURE_DIM * 2
    free = shutil.disk_usage(dataset_root).free
    reserve = int(minimum_free_gib * 1024**3)
    print(
        f"[dino capacity] pending_episodes={len(pending)} required_tib={required / 1024**4:.3f} "
        f"free_tib={free / 1024**4:.3f} reserve_gib={minimum_free_gib:.1f}",
        flush=True,
    )
    if required + reserve > free and not allow_low_disk:
        raise RuntimeError(
            "insufficient disk for exact BF16 DINO cache; add storage, select a smaller task subset, "
            "or explicitly pass --allow-low-disk after providing another safety mechanism"
        )


@torch.inference_mode()
def cache_dino(
    records,
    *,
    model_name: str,
    device: torch.device,
    batch_size: int,
    pyav_threads: int,
    overwrite: bool,
    rank: int,
    world_size: int,
) -> None:
    assigned = records[rank::world_size]
    pending = [
        record
        for record in assigned
        if overwrite
        or not valid_dino_cache(
            episode_dino_cache_path(record.task_dir, record.episode_index), record.length
        )
    ]
    print(f"[dino rank={rank}] assigned={len(assigned)} pending={len(pending)} device={device}", flush=True)
    if not pending:
        return
    encoder = DinoV3FeatureEncoder(model_name, device=device, dtype=torch.float32)
    started = time.monotonic()
    for position, record in enumerate(pending, start=1):
        views = decode_views(record.task_dir, record.episode_index, record.length, pyav_threads)
        view_features = []
        for key in ROBOTWIN_VIEW_KEYS:
            raw = torch.from_numpy(views[key]).permute(0, 3, 1, 2).contiguous()
            parts = []
            for start in range(0, record.length, batch_size):
                images = raw[start : start + batch_size].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                ).div_(255.0)
                parts.append(
                    encoder(images).to(device="cpu", dtype=torch.bfloat16).contiguous()
                )
            view_features.append(torch.cat(parts, dim=0))
        features = torch.cat(view_features, dim=1)
        expected = (record.length, *EXPECTED_DINO_SHAPE)
        if tuple(features.shape) != expected:
            raise RuntimeError(f"Unexpected DINO cache shape {tuple(features.shape)} != {expected}")
        atomic_torch_save(
            features,
            episode_dino_cache_path(record.task_dir, record.episode_index),
        )
        if position % 10 == 0 or position == len(pending):
            elapsed = time.monotonic() - started
            rate = position / max(elapsed, 1e-6)
            eta = (len(pending) - position) / max(rate, 1e-6)
            print(
                f"[dino rank={rank}] {position}/{len(pending)} rate={rate:.3f}ep/s eta={eta:.0f}s "
                f"task={record.task_dir.parent.name}/{record.task_dir.name} episode={record.episode_index}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-glob", action="append", default=[])
    parser.add_argument("--stage", choices=("all", "language", "images", "dino"), default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--language-batch-size", type=int, default=4)
    parser.add_argument("--dino-batch-size", type=int, default=64)
    parser.add_argument("--dino-model", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--minimum-free-gib", type=float, default=256.0)
    parser.add_argument("--allow-low-disk", action="store_true")
    parser.add_argument("--pyav-thread-count", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    task_globs = tuple(args.task_glob or DEFAULT_TASK_GLOBS)
    records = discover_lerobot_episode_records(root, task_globs)
    tasks = sorted({record.task_dir for record in records})
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
        selected = set(tasks)
        records = [record for record in records if record.task_dir in selected]
    if args.max_episodes > 0:
        records = records[: args.max_episodes]

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if rank == 0:
        print(
            f"tasks={len(tasks)} episodes={len(records)} task_globs={task_globs} world_size={world_size}",
            flush=True,
        )
        write_manifests(tasks, checkpoint, args.dino_model)
    if args.stage == "dino":
        # Every torchrun rank performs the same read-only guard before any rank
        # can start writing, so a rank-0 failure cannot race other workers.
        check_dino_cache_capacity(
            records,
            root,
            overwrite=args.overwrite,
            minimum_free_gib=args.minimum_free_gib,
            allow_low_disk=args.allow_low_disk,
        )
    if args.stage in ("all", "language") and rank == 0:
        cache_language(records, checkpoint, device, args.language_batch_size, args.overwrite)
    if args.stage in ("all", "images"):
        cache_images(
            records,
            checkpoint,
            device,
            args.batch_size,
            args.pyav_thread_count,
            args.overwrite,
            rank,
            world_size,
        )
    if args.stage == "dino":
        cache_dino(
            records,
            model_name=args.dino_model,
            device=device,
            batch_size=args.dino_batch_size,
            pyav_threads=args.pyav_thread_count,
            overwrite=args.overwrite,
            rank=rank,
            world_size=world_size,
        )


if __name__ == "__main__":
    main()
