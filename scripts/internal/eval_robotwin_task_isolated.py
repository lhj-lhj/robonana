#!/usr/bin/env python3
# 中文：内部辅助：按 episode 隔离仿真进程并执行单任务评测。
# English: Internal helper: evaluate one task with isolated episode processes.
# 调用 / Invocation: 由并行评测入口启动；写评测状态和结果。 / Spawned by the parallel evaluator; writes eval state and results.
# 导航 / Guide: scripts/README.md (internal)
"""Evaluate one RoboTwin task with one fresh simulator process per episode."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class AttemptMode:
    name: str
    oidn_device: str


class StopRequested(InterruptedError):
    pass


def _raise_stop_requested(signum: int, _frame: Any) -> None:
    raise StopRequested(f"received signal {signum}")


def initial_seed(seed_group: int) -> int:
    return 100_000 * (1 + seed_group)


def attempt_modes(gpu_attempts: int, cpu_fallback: bool) -> tuple[AttemptMode, ...]:
    if gpu_attempts < 1:
        raise ValueError("gpu_attempts must be positive")
    modes = [AttemptMode(f"oidn_cuda_{index + 1}", "cuda") for index in range(gpu_attempts)]
    if cpu_fallback:
        modes.append(AttemptMode("oidn_cpu_fallback", "cpu"))
    return tuple(modes)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def validate_episode_metadata(payload: dict[str, Any], expected_start_seed: int) -> dict[str, int]:
    required = {"start_seed", "accepted_seed", "next_seed", "success"}
    if set(payload) != required:
        raise ValueError(f"episode metadata keys must be {sorted(required)}, got {sorted(payload)}")
    normalized = {key: int(payload[key]) for key in required}
    if normalized["start_seed"] != expected_start_seed:
        raise ValueError(
            f"episode start seed {normalized['start_seed']} != expected {expected_start_seed}"
        )
    if normalized["next_seed"] <= normalized["start_seed"]:
        raise ValueError("episode metadata did not advance the seed")
    if normalized["accepted_seed"] != normalized["next_seed"] - 1:
        raise ValueError("accepted_seed must equal next_seed - 1")
    if normalized["success"] not in (0, 1):
        raise ValueError("single-episode success must be 0 or 1")
    return normalized


def read_ledger(path: Path, target_episodes: int, start_seed: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    expected_seed = start_seed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("episode_index", -1)) != len(rows):
                raise ValueError(f"non-contiguous episode index at {path}:{line_number}")
            metadata = validate_episode_metadata(
                {key: row[key] for key in ("start_seed", "accepted_seed", "next_seed", "success")},
                expected_seed,
            )
            expected_seed = metadata["next_seed"]
            rows.append(row)
    if len(rows) > target_episodes:
        raise ValueError(f"ledger has {len(rows)} rows but target is {target_episodes}")
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def swallowed_error_count(path: Path) -> int:
    """Count RoboTwin's opaque retry marker in the bounded log tail."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 2 * 1024 * 1024))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return 0
    return tail.count("error occurs !")


def newest_result_dir(root: Path, started_at: float) -> Path | None:
    if not root.is_dir():
        return None
    candidates = [
        path.parent
        for path in root.glob("*/_result.txt")
        if path.stat().st_mtime >= started_at - 1.0
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def write_outputs(
    output_dir: Path,
    task_name: str,
    rows: Iterable[dict[str, Any]],
    target_episodes: int,
) -> None:
    materialized = list(rows)
    if len(materialized) != target_episodes:
        return
    successes = sum(int(row["success"]) for row in materialized)
    rate = successes / target_episodes
    (output_dir / "results.csv").write_text(
        "task,success,total,success_rate\n"
        f"{task_name},{successes},{target_episodes},{rate:.10f}\n",
        encoding="utf-8",
    )
    videos = [str(row["video_path"]) for row in materialized if row.get("video_path")]
    (output_dir / "mp4_manifest.txt").write_text(
        "".join(f"{path}\n" for path in videos), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "episodes": target_episodes,
                "oidn_cpu_fallback_episodes": sum(
                    row.get("mode") == "oidn_cpu_fallback" for row in materialized
                ),
                "success": successes,
                "success_rate": rate,
                "task": task_name,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    # 中文：保留旧 CLI 和日志读取 helper；实际评测只调用统一入口。
    # Explicit retry seed replaces only the candidate start, never a completed ledger.
    if args.cpu_fallback:
        raise ValueError("CPU fallback changes rendering; unified eval requires CUDA OIDN")
    from eval_legacy_args import arguments
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_multitask_mbrl import main
    env = dict(os.environ, ROBONANA_EVAL_TASKS=args.task_name,
               ROBONANA_EVAL_RUN_DIR=str(args.output_dir),
               ROBONANA_EVAL_SEED_GROUP=str(args.seed_group),
               ROBONANA_EPISODE_TIMEOUT_SECONDS=str(args.episode_timeout_seconds),
               ROBONANA_EPISODE_GPU_ATTEMPTS=str(args.gpu_attempts))
    argv = arguments(args.task_config, args.test_num, env)
    if args.start_seed is not None:
        argv[argv.index('--seed-start')+1] = str(args.start_seed)
    sys.argv = [sys.argv[0], *argv]
    main()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--test-num", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--launch-client", type=Path, required=True)
    parser.add_argument("--ckpt-setting", default="fact")
    parser.add_argument("--seed-group", type=int, default=0)
    parser.add_argument('--start-seed',type=int,default=os.environ.get('ROBONANA_EVAL_START_SEED'),
                        help='Explicit candidate seed for historical failure reproduction')
    parser.add_argument("--episode-timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--max-swallowed-errors",
        type=int,
        default=int(os.environ.get("ROBONANA_MAX_SWALLOWED_ERRORS", "32")),
        help="Abort a hung RoboTwin retry loop after this many opaque errors; 0 disables.",
    )
    parser.add_argument("--gpu-attempts", type=int, default=2)
    parser.add_argument("--cpu-fallback", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if (
        args.test_num < 1
        or args.episode_timeout_seconds < 1
        or args.gpu_attempts < 1
        or args.max_swallowed_errors < 0
        or (args.start_seed is not None and args.start_seed < 0)
    ):
        parser.error("test-num, episode-timeout-seconds, and gpu-attempts must be valid")
    if not args.launch_client.is_file():
        parser.error(f"launch client does not exist: {args.launch_client}")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
