#!/usr/bin/env python3
"""Build the full Clean+Randomized RoboNana episode index and norm stats."""

from __future__ import annotations

import argparse

from robonana.data.stats import write_robotwin_lerobot_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--task-glob", action="append", default=[])
    parser.add_argument("--action-chunk", type=int, default=48)
    parser.add_argument("--action-dim", type=int, default=14)
    args = parser.parse_args()
    task_globs = tuple(args.task_glob or ("Clean/*", "Randomized/*"))
    index, stats = write_robotwin_lerobot_metadata(
        args.dataset_root,
        task_globs=task_globs,
        action_chunk=args.action_chunk,
        action_dim=args.action_dim,
    )
    print(f"index={index}")
    print(f"stats={stats}")


if __name__ == "__main__":
    main()
