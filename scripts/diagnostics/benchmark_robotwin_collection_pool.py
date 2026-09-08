#!/usr/bin/env python3
# 中文：诊断：测量 RLinf 持久化并行采集的吞吐。
# English: Diagnostic: measure RLinf persistent parallel collection throughput.
# 调用 / Invocation: 会启动服务与仿真并写测试轨迹；不是正式训练轮次入口。 / Starts services/simulators and writes probe rollouts; not a production round launcher.
# 导航 / Guide: scripts/README.md (diagnostics)
"""Benchmark RLinf-style persistent collectors with the existing policy server.

This is an opt-in infrastructure probe, not a replacement training/eval launcher.
Source HDF5s supply previously accepted environment seeds and exact instructions;
use only with the same deployed RoboTwin task config/assets as those episodes.
References are documented in robonana.sim.collection_pool (no RLinf trainer).
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import h5py

from robonana.sim.collection_pool import EpisodeQueue, validate_jobs
from robonana.normalization import A_STATS_PATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "internal"))
from eval_robotwin_task_isolated import terminate_process_group

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--sim-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--server-gpu", type=int, default=6)
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--robotwin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--initial-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8194)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--batch-wait-ms", type=float, default=0)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--inference-mode", choices=("action_only", "action_q_rejection"), default="action_q_rejection")
    opts = parser.parse_args()
    if not 1 <= opts.inference_batch_size <= 8 or not 1 <= opts.candidate_batch_size <= 32:
        parser.error("request batch must be 1..8 and candidate batch 1..32")
    if not 0 <= opts.batch_wait_ms <= 1000 or (opts.inference_batch_size > 1 and opts.batch_wait_ms == 0):
        parser.error("use a positive bounded wait (<=1000 ms) for dynamic multi-request batching")
    if any(gpu < 0 for gpu in opts.sim_gpus):
        parser.error("GPU ids must be nonnegative; repeat an id for multiple isolated workers")
    jobs, signatures = [], set()
    for path in opts.source_episodes:
        with h5py.File(path, "r") as handle:
            signatures.add((str(handle.attrs["task_name"]), str(handle.attrs["task_config"])))
            jobs.append({"seed": int(handle.attrs["seed"]),
                         "instruction": str(handle.attrs["instruction"]),
                         "source": str(path.resolve())})
    if len(signatures) != 1:
        parser.error("all seeds must belong to the same task/config")
    task_name, task_config = signatures.pop()
    validate_jobs(jobs, len(opts.sim_gpus))
    output = opts.output.resolve()
    output.mkdir(parents=True, exist_ok=False)  # Never resume/overwrite a probe.
    dataset = output / "dataset"
    queue_path = output / "episode_queue.sqlite"
    queue = EpisodeQueue(queue_path)
    queue.initialize(jobs)
    pythonpath = os.pathsep.join(str(ROOT / p) for p in
        ("src", "third_party/FACT", "third_party/flux2_official/src", "third_party/flux2/src"))
    common = dict(os.environ, PYTHONPATH=pythonpath, PYTHONUNBUFFERED="1")
    # Do not inherit optional diagnostics or global instruction overrides.
    common.update(ROBONANA_SELECTED_WORLD_ROOT="", ROBONANA_EVAL_INSTRUCTION="",
                  ROBONANA_OVERLAY_CHUNK_RETURN="0", ROBONANA_Q_DIAGNOSTICS_PATH="")
    server_env = dict(common, CUDA_VISIBLE_DEVICES=str(opts.server_gpu),
                      ROBONANA_REJECTION_CANDIDATE_BATCH_SIZE=str(opts.candidate_batch_size))
    server_cmd = [sys.executable, str(ROOT / "scripts/services/inference_server_robotwin_batched.py"),
        "--checkpoint", str(opts.checkpoint.resolve()), "--model-config", str(opts.model_config.resolve()),
        "--flux-checkpoint-dir", str(ROOT / "checkpoints/FLUX.2-klein-base-4B"),
        "--stats-path", str(A_STATS_PATH),
        "--model-device", "cuda:0", "--vae-device", "cuda:0", "--text-encoder-device", "cuda:0",
        "--inference-mode", opts.inference_mode, "--port", str(opts.port),
        # Scheduling knobs only; algorithm settings come from the checkpoint.
        "--max-batch-size", str(opts.inference_batch_size),
        "--max-batch-wait-ms", str(opts.batch_wait_ms), "--max-clients", "8",
        "--batch-metrics-path", str(output / "batch_metrics.jsonl")]
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(opts).items()}
    configuration["source_episodes"] = [str(p) for p in opts.source_episodes]
    configuration["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (output / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    (output / "seeds.json").write_text(json.dumps({"task_name": task_name,
        "task_config": task_config, "jobs": jobs}, indent=2), encoding="utf-8")
    children, logs, workers = [], [], []
    start = time.perf_counter()
    def interrupted(signum, _frame):
        raise RuntimeError(f"interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        logs.append((output / "server.log").open("w"))
        server = subprocess.Popen(server_cmd, env=server_env, cwd=ROOT, stdout=logs[-1],
                                  stderr=subprocess.STDOUT, start_new_session=True)
        children.append(server)
        for rank, gpu in enumerate(opts.sim_gpus):
            worker_dir = output / f"worker_{rank}_gpu_{gpu}"
            worker_dir.mkdir()
            job_path = worker_dir / "jobs.json"
            job_path.write_text(json.dumps({"task_name": task_name, "task_config": task_config,
                                           "jobs": jobs}), encoding="utf-8")
            runtime = worker_dir / "runtime"
            runtime.mkdir(mode=0o700)
            worker_env = dict(common, CUDA_VISIBLE_DEVICES=str(gpu), OIDN_DEFAULT_DEVICE="cuda",
                ROBONANA_SAPIEN_RENDER_DEVICE="cuda:0", ROBONANA_ROBOTWIN_STATIC_CAMERAS="head_camera",
                XDG_RUNTIME_DIR=str(runtime), FACT_ROBOTWIN_EVAL_VIDEO_LOG="0",
                ROBONANA_ROLLOUT_DATASET_ROOT=str(dataset),
                ROBONANA_INITIAL_DATASET_ROOT=str(opts.initial_dataset.resolve()),
                ROBONANA_ROLLOUT_CHECKPOINT=str(opts.checkpoint.resolve()))
            logs.append((worker_dir / "client.log").open("w"))
            # Never resolve a venv python symlink: invoking its target bypasses
            # pyvenv.cfg and silently selects the wrong SAPIEN dependency set.
            worker = subprocess.Popen([str(opts.sim_python.absolute()),
                str(ROOT / "scripts/internal/collect_robotwin_pool_worker.py"), "--jobs", str(job_path),
                "--robotwin", str(opts.robotwin.resolve()), "--output", str(worker_dir),
                "--vector-env-checkout", str(ROOT / "third_party/RoboTwin_RLinf"),
                "--queue", str(queue_path), "--worker-id", str(rank),
                "--port", str(opts.port)], cwd=ROOT, env=worker_env, stdout=logs[-1],
                stderr=subprocess.STDOUT, start_new_session=True)
            children.append(worker)
            workers.append(worker)
        with (output / "gpu_usage.jsonl").open("w", buffering=1) as gpu_log:
            while any(worker.poll() is None for worker in workers):
                if server.poll() is not None or any(worker.poll() not in (None, 0) for worker in workers):
                    raise RuntimeError("server/worker failed; inspect independent logs")
                if time.perf_counter() - start > opts.timeout_seconds:
                    raise TimeoutError("collection probe exceeded its bounded deadline")
                sample = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                gpu_log.write(json.dumps({"elapsed": time.perf_counter() - start, "gpus": sample.stdout}) + "\n")
                time.sleep(2)
        if any(worker.returncode != 0 for worker in workers):
            raise RuntimeError("worker exited with failure")
        elapsed = time.perf_counter() - start
        rows = []
        for complete in sorted(output.glob("worker_*/complete.json")):
            rows.extend(json.loads(complete.read_text()))
        actual = []
        for file in dataset.glob("*/robonana_rollout/data/episode*.hdf5"):
            with h5py.File(file, "r") as handle:
                frames = len(handle["joint_action/vector"])
                assert bool(handle.attrs["has_final_observation"])
                assert not bool(handle["transition_valid"][-1])
                assert all(bool(v) for v in handle["transition_valid"][:-1])
                for camera in ("head_camera", "left_camera", "right_camera"):
                    assert len(handle[f"observation/{camera}/rgb"]) == frames
                actual.append(int(handle.attrs["seed"]))
        expected = sorted(int(job["seed"]) for job in jobs)
        if sorted(actual) != expected or sorted(r["seed"] for r in rows) != expected:
            raise RuntimeError("completed ledger/HDF5 seeds disagree with assigned jobs")
        if queue.counts() != {"done": len(jobs)}:
            raise RuntimeError("queue has unfinished claims")
        summary = {"episodes": rows, "wall_seconds_including_startup": elapsed,
                   "episodes_per_hour": len(rows) * 3600 / elapsed,
                   "dataset_validated": True, "inference_batch_size": opts.inference_batch_size,
                   "candidate_batch_size": opts.candidate_batch_size,
                   "queue_counts": queue.counts(), "workers": len(workers)}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary), flush=True)
    finally:
        for child in reversed(children):
            terminate_process_group(child, grace_seconds=10)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
