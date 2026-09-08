#!/usr/bin/env python3
# 中文：数据维护：生成原始 LeRobot 数据的索引与统计来源信息。
# English: Data maintenance: build original LeRobot metadata and normalization provenance.
# 调用 / Invocation: 非日常训练入口；会写元数据，勿对已有 A 统计随意重建。 / Not a daily training entry; writes metadata, do not casually rebuild existing A statistics.
# 导航 / Guide: scripts/README.md (data)
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
