#!/usr/bin/env python3
"""Validate full FACT RoboTwin-v2 metadata and RoboNana FLUX/Qwen caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from robonana.data.flux_cache import episode_cache_path, episode_language_context_path
from robonana.data.robotwin_lerobot import load_lerobot_episode_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-clean", type=int, default=2500)
    parser.add_argument("--expected-randomized", type=int, default=25000)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    index_path = root / "robonana_index.json"
    stats_path = root / "robonana_norm_stats.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    records = load_lerobot_episode_records(root, ("Clean/*", "Randomized/*"), index_path)
    clean = sum(record.task_dir.parent.name == "Clean" for record in records)
    randomized = sum(record.task_dir.parent.name == "Randomized" for record in records)
    if (clean, randomized) != (args.expected_clean, args.expected_randomized):
        raise RuntimeError(f"episode counts {(clean, randomized)} do not match expected")
    if index.get("source_format") != "lerobot-v2" or stats.get("source_format") != "lerobot-v2":
        raise RuntimeError("metadata source format mismatch")
    missing = []
    bad = []
    for record in records:
        image_path = episode_cache_path(record.task_dir, record.episode_index)
        language_path = episode_language_context_path(record.task_dir, record.episode_index)
        if not image_path.is_file() or not language_path.is_file():
            missing.append((str(image_path), str(language_path)))
            continue
        try:
            image = torch.load(image_path, map_location="cpu", weights_only=True, mmap=True)
            language = torch.load(language_path, map_location="cpu", weights_only=True, mmap=True)
            if tuple(image.shape) != (record.length, 288, 128) or image.dtype != torch.bfloat16:
                bad.append((str(image_path), tuple(image.shape), str(image.dtype)))
            if tuple(language.shape) != (512, 7680) or language.dtype != torch.bfloat16:
                bad.append((str(language_path), tuple(language.shape), str(language.dtype)))
        except Exception as error:
            bad.append((str(image_path), repr(error)))
    if missing or bad:
        raise RuntimeError(f"cache validation failed: missing={len(missing)} bad={len(bad)} samples={missing[:2] + bad[:2]}")
    print(
        json.dumps(
            {
                "tasks": len({record.task_dir for record in records}),
                "episodes": len(records),
                "clean": clean,
                "randomized": randomized,
                "image_caches": len(records),
                "language_caches": len(records),
                "status": "complete",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
