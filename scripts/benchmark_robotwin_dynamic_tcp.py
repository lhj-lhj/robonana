#!/usr/bin/env python3
"""Stress the real RoboNana policy through concurrent persistent FACT clients."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import threading
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for upstream in reversed(
    (
        REPO_ROOT / "src",
        REPO_ROOT / "third_party" / "FACT",
        REPO_ROOT / "third_party" / "flux2" / "src",
        REPO_ROOT / "third_party" / "flux2_official" / "src",
    )
):
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

from robonana.data.robotwin_lerobot import (  # noqa: E402
    RoboTwinLeRobotDataset,
    lerobot_episode_instruction,
)
from robonana.inference.batched_policy import BatchedRoboNanaRobotWinPolicy  # noqa: E402
from robonana.inference.dynamic_batch_server import (  # noqa: E402
    DynamicBatchRobotInferenceServer,
)
from world_action_model import apply_runtime_compat  # noqa: E402
from world_action_model.sockets import RobotInferenceClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--flux-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--requests-per-client", type=int, default=3)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-wait-ms", type=float, default=6.0)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--vae-device", default="cuda:0")
    parser.add_argument("--text-encoder-device", default="cuda:0")
    return parser.parse_args()


def first_observation(dataset: RoboTwinLeRobotDataset, record) -> dict[str, object]:
    states, _ = dataset._episode_state_action(record)
    return {
        "observation.state": torch.from_numpy(states[0].copy()),
        **dataset._future_dino_images(record, 0),
        "instruction": lerobot_episode_instruction(record.task_dir, record.episode_index),
    }


def distinct_task_observations(
    dataset: RoboTwinLeRobotDataset, count: int
) -> list[dict[str, object]]:
    selected = []
    seen = set()
    for record in dataset.records:
        if record.task_name in seen:
            continue
        selected.append(first_observation(dataset, record))
        seen.add(record.task_name)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"dataset has only {len(selected)} distinct tasks, need {count}")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> int:
    apply_runtime_compat()
    args = parse_args()
    if min(args.clients, args.requests_per_client, args.max_batch_size) < 1:
        raise ValueError("clients, requests-per-client, and max-batch-size must be positive")

    dataset = RoboTwinLeRobotDataset(
        str(args.dataset_root.expanduser().resolve()),
        stats_path=str(args.stats_path.expanduser().resolve()),
        index_path=(
            None if args.index_path is None else str(args.index_path.expanduser().resolve())
        ),
        task_globs=("Clean/*", "Randomized/*"),
        action_chunk=48,
        action_dim=14,
        max_horizon=48,
    )
    dataset.open()
    observations = distinct_task_observations(dataset, args.clients)
    dataset.close()

    policy = BatchedRoboNanaRobotWinPolicy(
        checkpoint=args.checkpoint.expanduser().resolve(),
        model_config=args.model_config.expanduser().resolve(),
        flux_checkpoint_dir=args.flux_checkpoint_dir.expanduser().resolve(),
        stats_path=args.stats_path.expanduser().resolve(),
        model_device=args.model_device,
        vae_device=args.vae_device,
        text_encoder_device=args.text_encoder_device,
        dtype=torch.bfloat16,
        action_chunk=48,
        horizon=24,
        num_inference_steps=args.num_inference_steps,
    )
    server = DynamicBatchRobotInferenceServer(
        policy,
        host="127.0.0.1",
        port=0,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_batch_wait_ms,
        max_clients=args.clients + 2,
    )
    port = server.server_socket.getsockname()[1]
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    barrier = threading.Barrier(args.clients)
    latencies: list[float] = []
    timing_rows: list[dict[str, float]] = []
    result_lock = threading.Lock()

    device = torch.device(args.model_device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_bytes = torch.cuda.memory_allocated(device)
    else:
        baseline_bytes = 0

    def run_client(client_index: int) -> int:
        client = RobotInferenceClient(host="127.0.0.1", port=port, timeout_ms=600_000)
        completed = 0
        try:
            barrier.wait(timeout=30)
            for request_index in range(args.requests_per_client):
                sampling_seed = 202608290000 + client_index * 10_000 + request_index
                request = dict(observations[client_index])
                request["sampling_seed"] = sampling_seed
                started = time.perf_counter()
                response = client.inference(request)
                elapsed = time.perf_counter() - started
                action = torch.as_tensor(response["action"])
                if tuple(action.shape) != (48, 14) or not torch.isfinite(action).all():
                    raise RuntimeError(f"invalid action response {tuple(action.shape)}")
                if int(response["_sampling_seed"]) != sampling_seed:
                    raise RuntimeError("response sampling seed was cross-wired between clients")
                with result_lock:
                    latencies.append(elapsed)
                    timing_rows.append(dict(response["_policy_timing_ms"]))
                completed += 1
            return completed
        finally:
            client.close()

    wall_started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.clients) as executor:
            futures = [executor.submit(run_client, index) for index in range(args.clients)]
            completed = sum(future.result(timeout=1800) for future in futures)
    finally:
        stop_client = RobotInferenceClient(host="127.0.0.1", port=port, timeout_ms=5000)
        try:
            stop_client.kill_server()
        finally:
            stop_client.close()
        server_thread.join(timeout=10)
    wall_elapsed = time.perf_counter() - wall_started

    if server_thread.is_alive():
        raise RuntimeError("dynamic TCP server did not shut down cleanly")
    expected = args.clients * args.requests_per_client
    if completed != expected or len(latencies) != expected:
        raise RuntimeError(f"completed {completed}/{expected} requests")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
    else:
        peak_bytes = 0

    response_batch_sizes = Counter(int(row["batch_size"]) for row in timing_rows)
    batch_invocations = sum(count // size for size, count in response_batch_sizes.items())
    report = {
        "clients": args.clients,
        "requests_per_client": args.requests_per_client,
        "completed_requests": completed,
        "max_batch_size": args.max_batch_size,
        "max_batch_wait_ms": args.max_batch_wait_ms,
        "wall_seconds": wall_elapsed,
        "requests_per_second": completed / wall_elapsed,
        "latency_seconds": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "response_batch_size_counts": dict(sorted(response_batch_sizes.items())),
        "batch_invocations": batch_invocations,
        "baseline_allocated_gib": baseline_bytes / 2**30,
        "peak_allocated_gib": peak_bytes / 2**30,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
