#!/usr/bin/env python3
"""Generate the portable episode index and FACT-style normalization stats."""

from __future__ import annotations

import argparse
from pathlib import Path

from robonana.data.stats import write_robotwin_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task-glob", default="*/aloha-agilex_clean_50")
    parser.add_argument("--action-chunk", type=int, default=48)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--stats-path", type=Path)
    args = parser.parse_args()
    index_path, stats_path = write_robotwin_metadata(
        args.dataset_root,
        task_glob=args.task_glob,
        action_chunk=args.action_chunk,
        action_dim=args.action_dim,
        index_path=args.index_path,
        stats_path=args.stats_path,
    )
    print(f"episode index: {index_path}")
    print(f"normalization stats: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
