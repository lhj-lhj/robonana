#!/usr/bin/env python3
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
    signal.signal(signal.SIGTERM, _raise_stop_requested)
    signal.signal(signal.SIGINT, _raise_stop_requested)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "episodes.jsonl"
    seed_start = initial_seed(args.seed_group)
    rows = read_ledger(ledger_path, args.test_num, seed_start)
    if rows:
        print(f"[resume] {args.task_name}: {len(rows)}/{args.test_num} episodes", flush=True)
    modes = attempt_modes(args.gpu_attempts, args.cpu_fallback)
    eval_root = (
        Path(os.environ["ROBOTWIN_PATH"])
        / "eval_result"
        / args.task_name
        / os.environ.get("POLICY_NAME", "robonana_robotwin.adapter")
        / args.task_config
        / args.ckpt_setting
    )

    for episode_index in range(len(rows), args.test_num):
        start_seed = seed_start if not rows else int(rows[-1]["next_seed"])
        episode_dir = output_dir / "episodes" / f"episode_{episode_index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        completed_row: dict[str, Any] | None = None

        for attempt_index, mode in enumerate(modes, start=1):
            attempt_dir = episode_dir / f"attempt_{attempt_index}_{mode.name}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            runtime_dir = attempt_dir / "runtime"
            runtime_dir.mkdir(exist_ok=True)
            metadata_path = attempt_dir / "metadata.json"
            log_path = attempt_dir / "client.log"
            metadata_path.unlink(missing_ok=True)
            environment = os.environ.copy()
            environment.update(
                {
                    "OIDN_DEFAULT_DEVICE": mode.oidn_device,
                    "PYTHONUNBUFFERED": "1",
                    "ROBONANA_EVAL_EPISODE_METADATA": str(metadata_path),
                    "ROBONANA_EVAL_START_SEED": str(start_seed),
                    "TEST_NUM": "1",
                    "XDG_RUNTIME_DIR": str(runtime_dir),
                }
            )
            command = [
                "bash",
                str(args.launch_client),
                args.task_name,
                args.task_config,
                args.ckpt_setting,
                str(args.seed_group),
            ]
            started_at = time.time()
            started_monotonic = time.monotonic()
            return_code: int | None = None
            timed_out = False
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                log.write(
                    f"mode={mode.name} start_seed={start_seed} "
                    f"timeout={args.episode_timeout_seconds}s\n"
                )
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    return_code = process.wait(timeout=args.episode_timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminate_process_group(process)
                    return_code = process.returncode
                except StopRequested:
                    terminate_process_group(process)
                    raise

            duration = time.monotonic() - started_monotonic
            error: str | None = None
            metadata: dict[str, int] | None = None
            if timed_out:
                error = f"episode exceeded {args.episode_timeout_seconds}s"
            elif return_code != 0:
                error = f"client exited with rc={return_code}"
            elif not metadata_path.is_file():
                error = "client succeeded without episode metadata"
            else:
                try:
                    metadata = validate_episode_metadata(read_json(metadata_path), start_seed)
                except (KeyError, TypeError, ValueError) as exc:
                    error = f"invalid episode metadata: {exc}"

            attempt_record = {
                "attempt": attempt_index,
                "duration_seconds": round(duration, 3),
                "error": error,
                "log": str(log_path),
                "mode": mode.name,
                "return_code": return_code,
                "start_seed": start_seed,
                "timed_out": timed_out,
            }
            append_jsonl(output_dir / "attempts.jsonl", attempt_record)
            if error is not None:
                print(
                    f"[retry] {args.task_name} episode={episode_index} "
                    f"mode={mode.name}: {error}",
                    flush=True,
                )
                continue

            assert metadata is not None
            result_dir = newest_result_dir(eval_root, started_at)
            video_path: str | None = None
            if os.environ.get("EVAL_VIDEO_LOG", "1") != "0":
                candidate = result_dir / "episode0.mp4" if result_dir is not None else None
                if candidate is None or not candidate.is_file() or candidate.stat().st_size == 0:
                    print(
                        f"[retry] {args.task_name} episode={episode_index} "
                        f"mode={mode.name}: missing completed MP4",
                        flush=True,
                    )
                    continue
                video_path = str(candidate.resolve())

            completed_row = {
                **metadata,
                "attempt": attempt_index,
                "duration_seconds": round(duration, 3),
                "episode_index": episode_index,
                "log": str(log_path),
                "mode": mode.name,
                "result_dir": str(result_dir.resolve()) if result_dir is not None else None,
                "video_path": video_path,
            }
            append_jsonl(ledger_path, completed_row)
            rows.append(completed_row)
            write_outputs(output_dir, args.task_name, rows, args.test_num)
            print(
                f"[episode] {args.task_name} {episode_index + 1}/{args.test_num} "
                f"seed={metadata['accepted_seed']} success={metadata['success']} "
                f"mode={mode.name} duration={duration:.1f}s",
                flush=True,
            )
            break

        if completed_row is None:
            (output_dir / "FAILED").write_text(
                f"episode={episode_index} start_seed={start_seed}\n", encoding="utf-8"
            )
            print(
                f"[failed] {args.task_name}: episode={episode_index} "
                f"start_seed={start_seed} exhausted {len(modes)} attempts",
                file=sys.stderr,
                flush=True,
            )
            return 1

    (output_dir / "FAILED").unlink(missing_ok=True)
    write_outputs(output_dir, args.task_name, rows, args.test_num)
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
    parser.add_argument("--episode-timeout-seconds", type=int, default=3600)
    parser.add_argument("--gpu-attempts", type=int, default=2)
    parser.add_argument("--cpu-fallback", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.test_num < 1 or args.episode_timeout_seconds < 1 or args.gpu_attempts < 1:
        parser.error("test-num, episode-timeout-seconds, and gpu-attempts must be positive")
    if not args.launch_client.is_file():
        parser.error(f"launch client does not exist: {args.launch_client}")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
