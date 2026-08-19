#!/usr/bin/env python3
"""Cache Qwen3 language context and FLUX.2 AE tokens for raw RoboTwin data.

The script deliberately reuses FACT's public three-view layout helper and the
official FLUX.2 Qwen3 forward method.  With ``torchrun`` each local rank owns
one GPU and processes a disjoint subset of episodes; no process group is
needed because cache files are independent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
for upstream in (
    REPO_ROOT / "third_party" / "FACT",
    REPO_ROOT / "third_party" / "flux2",
    REPO_ROOT / "third_party" / "flux2" / "src",
    REPO_ROOT / "third_party" / "flux2_official" / "src",
):
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

from flux2.text_encoder import MAX_LENGTH, OUTPUT_LAYERS_QWEN3
from robonana.data.flux_cache import (
    CACHE_SCHEMA_VERSION,
    canonical_instruction,
    episode_cache_path,
    language_context_path,
)
from robonana.encoding import LocalQwen3Embedder, encode_flux2_image_tokens
from world_action_model.image_layouts import ROBOTWIN_VIEW_KEYS, build_robotwin_three_view_tensor


MAIN_VIEW_SIZE = (256, 192)
CANVAS_SIZE = (384, 192)
EXPECTED_IMAGE_TOKENS = 12 * 24
EXPECTED_LATENT_CHANNELS = 128
HDF5_CAMERAS = ("head_camera", "left_camera", "right_camera")


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def discover_task_dirs(dataset_root: Path, task_glob: str, max_tasks: int) -> list[Path]:
    tasks = sorted(path for path in dataset_root.glob(task_glob) if any((path / "data").glob("episode*.hdf5")))
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise FileNotFoundError(f"No raw RoboTwin tasks matching {task_glob!r} under {dataset_root}")
    return tasks


def episode_index(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"Unexpected RoboTwin episode filename: {path}")
    return int(match.group(1))


def write_manifest(task_dir: Path, checkpoint: Path) -> None:
    atomic_json_save(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_format": "RoboTwin raw HDF5",
            "layout": {
                "head_camera": [256, 192],
                "left_camera": [128, 96],
                "right_camera": [128, 96],
                "canvas": list(CANVAS_SIZE),
            },
            "language": {
                "encoder": "Qwen3-4B",
                "output_layers": list(OUTPUT_LAYERS_QWEN3),
                "max_length": MAX_LENGTH,
                "shape": [MAX_LENGTH, 7680],
                "dtype": "bfloat16",
            },
            "image": {
                "encoder": "FLUX.2 AutoencoderKLFlux2",
                "shape_per_frame": [EXPECTED_IMAGE_TOKENS, EXPECTED_LATENT_CHANNELS],
                "dtype": "bfloat16",
                "current_latent": "frame_latents[current_index]",
                "future_latent": "frame_latents[min(current_index + idx_h, episode_length - 1)]",
            },
            "checkpoint": str(checkpoint),
        },
        task_dir / "flux_cache" / "_manifest.json",
    )


@torch.inference_mode()
def cache_language(tasks: Iterable[Path], checkpoint: Path, device: torch.device, overwrite: bool) -> None:
    pending = [task for task in tasks if overwrite or not language_context_path(task).is_file()]
    if not pending:
        print("[language] all contexts already exist", flush=True)
        return
    print(f"[language] loading Qwen3 from {checkpoint / 'text_encoder'} on {device}", flush=True)
    embedder = LocalQwen3Embedder(checkpoint, device)
    for task_dir in pending:
        prompt, source_path = canonical_instruction(task_dir)
        context = embedder([prompt])[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        if tuple(context.shape) != (MAX_LENGTH, 7680):
            raise RuntimeError(f"Unexpected Qwen3 context shape for {task_dir}: {tuple(context.shape)}")
        atomic_torch_save(context, language_context_path(task_dir))
        atomic_json_save(
            {
                "prompt": prompt,
                "source": str(source_path) if source_path is not None else None,
                "shape": list(context.shape),
                "dtype": str(context.dtype).removeprefix("torch."),
            },
            task_dir / "flux_cache" / "language_context.json",
        )
        print(f"[language] {task_dir.parent.name}: {prompt}", flush=True)


def decode_rgb_batch(dataset: h5py.Dataset, start: int, stop: int) -> torch.Tensor:
    frames = []
    for encoded in dataset[start:stop]:
        with Image.open(BytesIO(bytes(encoded))) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        frames.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(frames).to(dtype=torch.float32).div_(255.0)


def build_composite_batch(handle: h5py.File, start: int, stop: int) -> torch.Tensor:
    views = {
        view_key: decode_rgb_batch(handle[f"observation/{camera}/rgb"], start, stop)
        for view_key, camera in zip(ROBOTWIN_VIEW_KEYS, HDF5_CAMERAS, strict=True)
    }
    composite = build_robotwin_three_view_tensor(views, main_dst_size=MAIN_VIEW_SIZE)
    if tuple(composite.shape[-2:]) != (CANVAS_SIZE[1], CANVAS_SIZE[0]):
        raise RuntimeError(f"FACT layout returned unexpected composite shape: {tuple(composite.shape)}")
    return composite.mul(2.0).sub(1.0)


@torch.inference_mode()
def encode_episode(vae, source: Path, output: Path, device: torch.device, batch_size: int) -> tuple[int, tuple[int, ...]]:
    token_batches = []
    with h5py.File(source, "r") as handle:
        lengths = [len(handle[f"observation/{camera}/rgb"]) for camera in HDF5_CAMERAS]
        if len(set(lengths)) != 1:
            raise RuntimeError(f"Camera lengths disagree in {source}: {dict(zip(HDF5_CAMERAS, lengths))}")
        episode_length = lengths[0]
        for start in range(0, episode_length, batch_size):
            stop = min(start + batch_size, episode_length)
            images = build_composite_batch(handle, start, stop).to(device=device, dtype=next(vae.parameters()).dtype)
            tokens = encode_flux2_image_tokens(vae, images)
            token_batches.append(tokens.to(device="cpu", dtype=torch.bfloat16).contiguous())
    frame_latents = torch.cat(token_batches, dim=0)
    expected = (episode_length, EXPECTED_IMAGE_TOKENS, EXPECTED_LATENT_CHANNELS)
    if tuple(frame_latents.shape) != expected:
        raise RuntimeError(f"Unexpected FLUX frame cache shape for {source}: {tuple(frame_latents.shape)} != {expected}")
    atomic_torch_save(frame_latents, output)
    return episode_length, tuple(frame_latents.shape)


def valid_episode_cache(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return value.ndim == 3 and tuple(value.shape[1:]) == (EXPECTED_IMAGE_TOKENS, EXPECTED_LATENT_CHANNELS)


def cache_images(
    tasks: list[Path],
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    max_episodes_per_task: int,
    overwrite: bool,
    rank: int,
    world_size: int,
) -> None:
    from diffusers.models import AutoencoderKLFlux2

    episodes: list[tuple[Path, Path, int]] = []
    for task_dir in tasks:
        paths = sorted((task_dir / "data").glob("episode*.hdf5"), key=episode_index)
        if max_episodes_per_task > 0:
            paths = paths[:max_episodes_per_task]
        episodes.extend((task_dir, source, episode_index(source)) for source in paths)
    assigned = episodes[rank::world_size]
    pending = [row for row in assigned if overwrite or not valid_episode_cache(episode_cache_path(row[0], row[2]))]
    print(
        f"[images rank={rank}] assigned={len(assigned)} pending={len(pending)} device={device}",
        flush=True,
    )
    if not pending:
        return
    vae = AutoencoderKLFlux2.from_pretrained(
        checkpoint,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    vae.requires_grad_(False)
    vae.to(device)
    for position, (task_dir, source, index) in enumerate(pending, start=1):
        output = episode_cache_path(task_dir, index)
        length, shape = encode_episode(vae, source, output, device, batch_size)
        print(
            f"[images rank={rank} {position}/{len(pending)}] {task_dir.parent.name}/episode{index} "
            f"frames={length} shape={shape}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "language", "images"), default="all")
    parser.add_argument("--task-glob", default="*/aloha-agilex_clean_50")
    parser.add_argument("--device", default=None, help="Defaults to cuda:LOCAL_RANK under torchrun, otherwise cuda:0.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-episodes-per-task", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    device = torch.device(args.device or (f"cuda:{rank}" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tasks = discover_task_dirs(args.dataset_root, args.task_glob, args.max_tasks)
    if rank == 0:
        for task_dir in tasks:
            write_manifest(task_dir, args.checkpoint)
        print(f"Discovered {len(tasks)} tasks under {args.dataset_root}", flush=True)
    if args.stage in ("all", "language") and rank == 0:
        cache_language(tasks, args.checkpoint, device, args.overwrite)
    if args.stage in ("all", "images"):
        cache_images(
            tasks,
            args.checkpoint,
            device,
            args.batch_size,
            args.max_episodes_per_task,
            args.overwrite,
            rank,
            world_size,
        )


if __name__ == "__main__":
    main()
