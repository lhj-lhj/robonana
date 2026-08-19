#!/usr/bin/env python3
"""Load one real RoboTwin batch through FACT's collator and report shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from fact_datasets import DefaultCollator
from robonana.data.robotwin_hdf5 import RoboTwinHDF5Dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--fixed-horizon", type=int, default=48)
    args = parser.parse_args()
    dataset = RoboTwinHDF5Dataset(
        str(args.dataset_root),
        stats_path=str(args.dataset_root / "robonana_norm_stats.json"),
        index_path=str(args.dataset_root / "robonana_index.json"),
        fixed_horizon=args.fixed_horizon,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=DefaultCollator(is_equal=True),
    )
    batch = next(iter(loader))
    shapes = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in batch.items()
        if hasattr(value, "shape")
    }
    expected = {
        "context": [1, 512, 7680],
        "current_latents": [1, 288, 128],
        "future_latents": [1, 288, 128],
        "state": [1, 14],
        "action": [1, 48, 14],
        "future_state": [1, 14],
        "value": [1, 1],
        "sample_index": [1],
    }
    for key, shape in expected.items():
        if shapes[key]["shape"] != shape:
            raise RuntimeError(f"unexpected {key} shape: {shapes[key]['shape']} != {shape}")
    if any(key.startswith("eval_") for key in batch):
        raise RuntimeError("ordinary training batches must not carry periodic eval targets")
    eval_future = dataset.load_eval_future_latents(
        int(batch["sample_index"][0].item()), dataset.eval_horizons
    )
    if list(eval_future.shape) != [3, 288, 128]:
        raise RuntimeError(f"unexpected lazy eval future shape: {list(eval_future.shape)}")
    print(
        json.dumps(
            {
                "dataset_length": len(dataset),
                "batch": shapes,
                "lazy_eval_future_latents": list(eval_future.shape),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
