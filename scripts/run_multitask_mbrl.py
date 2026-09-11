#!/usr/bin/env python3
# 中文：正式入口：50任务协议的配置、训练、失败采集和配对评测。
# English: Public entry: configure/train/collect/evaluate the fixed 50-task protocol.
# 调用 / Invocation: 默认只展示计划；--execute 才启动训练或仿真。 / Dry-run by default; --execute launches jobs.
"""See docs/MULTITASK_MBRL_PROTOCOL.md. No automatic unrelated job cancellation."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/diagnostics"), str(ROOT / "scripts/internal")]
STOP = threading.Event()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_bounded(command, *, env, log, timeout):
    """中文：给每个候选 seed 独立墙钟上限，清理自己启动的进程组。
    English: Reuse the isolated evaluator's process-group cleanup, including SIGTERM
    grace for the collector to close its separately-sessioned children.
    """
    from eval_robotwin_task_isolated import terminate_process_group
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            while not STOP.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return 124
                try:
                    return child.wait(timeout=min(.5, remaining))
                except subprocess.TimeoutExpired:
                    pass
            raise InterruptedError("Collection cancelled; do not replace this seed")
        finally:
            terminate_process_group(child, grace_seconds=40)


def training(opts):
    # FACT config entry reuses the existing loader, optimizer, scheduler and trainer.
    os.environ["ROBONANA_PROTOCOL_PHASE"] = opts.phase
    os.environ["ROBONANA_PROTOCOL_ROOT"] = str(opts.output.resolve())
    for flag, variable in (("checkpoint", "ROBONANA_MAC_PRETRAIN_CHECKPOINT"),
                           ("model_config", "ROBONANA_MAC_PRETRAIN_CONFIG"),
                           ("replay_root", "ROBONANA_REPLAY_ROOT")):
        if value := getattr(opts, flag):
            os.environ[variable] = str(value.resolve())
    os.environ["ROBONANA_GRADIENT_CHECKPOINTING"] = "1" if opts.gradient_checkpointing else "0"
    if opts.smoke_steps:
        os.environ["ROBONANA_PROTOCOL_SMOKE_STEPS"] = str(opts.smoke_steps)
    else:
        os.environ.pop("ROBONANA_PROTOCOL_SMOKE_STEPS", None)
    if opts.smoke_task_globs:
        os.environ["ROBONANA_PROTOCOL_SMOKE_TASK_GLOBS"] = opts.smoke_task_globs
    else:
        os.environ.pop("ROBONANA_PROTOCOL_SMOKE_TASK_GLOBS", None)
    from robonana.configs.multitask_mbrl import config, MILESTONES
    if opts.command == "aliases":
        # No duplicate multi-GB weights. Resolve only actual complete inference exports.
        from robonana.inference_contract import read_contract
        for step in MILESTONES[opts.phase]:
            matches = list((Path(config["project_dir"]) / "models").glob(f"checkpoint_*_step_{step}"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected exactly one complete milestone {step}: {matches}")
            checkpoint = matches[0].resolve()
            contract = read_contract(checkpoint / "transformer/diffusion_pytorch_model.bin")
            if contract["step"] != step:
                raise ValueError("Milestone contract step mismatch")
            name = f"base_ckpt_{step//1000}k" if opts.phase == "pretrain" else f"{opts.phase}_{step//1000}k_ckpt_0"
            destination = opts.output / name
            if opts.execute and not destination.exists():
                destination.symlink_to(checkpoint, target_is_directory=True)
            print(f"{destination} -> {checkpoint}")
        if opts.phase == "stage2" and opts.execute:
            destination = opts.output / "loop_ckpt_0"
            if not destination.exists():
                destination.symlink_to(checkpoint, target_is_directory=True)
        return
    print(json.dumps(config, indent=2, default=str))
    if not opts.execute:
        return
    project = Path(config["project_dir"])
    if project.exists():
        raise FileExistsError(f"Fresh-phase output exists; no implicit overwrite/resume: {project}")
    project.mkdir(parents=True)
    atomic_json(project / "protocol_config.json", config)
    env = dict(os.environ, ROBONANA_PROJECT_DIR=str(project), ROBONANA_PYTHON=sys.executable)
    env.setdefault("NCCL_NVLS_ENABLE", "0")  # Preserve NVLink P2P; do not set NCCL_P2P_DISABLE.
    subprocess.run(["bash", "scripts/run_robotwin_train.sh", "--config",
                    "robonana.configs.multitask_mbrl.config"], cwd=ROOT, env=env, check=True)


def collect_lane(opts, pairs, lane):
    from benchmark_robotwin_collection_pool import server_command
    from eval_robotwin_task_isolated import terminate_process_group
    from robonana.inference_contract import sha256_file
    server_opts = copy(opts)
    server_opts.port = opts.port + lane
    server_opts.inference_mode = "action_only" if opts.command == "collect" else "action_q_rejection"
    server_opts.inference_batch_size, server_opts.batch_wait_ms = 1, 0
    lane_root = opts.output / f"lane_{lane}"
    lane_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(opts.gpus[lane]),
               ROBONANA_REJECTION_CANDIDATE_BATCH_SIZE="32", FACT_ROBOTWIN_EVAL_VIDEO_LOG="0")
    sim_gpu = opts.gpus[lane + len(opts.gpus)//2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=opts.robotwin, text=True).strip()
    with (lane_root / "server.log").open("w") as log:
        server = subprocess.Popen(server_command(server_opts, lane_root), cwd=ROOT, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for task, task_config in pairs:
                task_root = opts.output / task / task_config
                task_root.mkdir(parents=True, exist_ok=True)
                ledger_path = task_root / "ledger.json"
                ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
                if any(row.get("result", {}).get("replay_verified") is False for row in ledger):
                    raise RuntimeError("Unresolved scout/replay mismatch in existing ledger; no automatic scene filtering")
                accepted = [r["job"] for r in ledger if r["status"] == "evaluated"]
                locked = None
                if opts.command == "eval":
                    locked = json.loads((opts.manifests / task / task_config / "seeds.json").read_text())
                    if locked["robotwin_commit"] != revision or locked["task_config_sha256"] != sha256_file(opts.robotwin / "task_config" / f"{task_config}.yml"):
                        raise ValueError("Locked scenes do not match simulator revision/config")
                attempts = len(ledger)
                while (attempts < len(locked["jobs"]) if locked else len(accepted) < opts.episodes):
                    if STOP.is_set():
                        raise InterruptedError("Collection cancelled")
                    if attempts >= opts.episodes * opts.candidate_multiplier:
                        raise RuntimeError("Candidate budget exhausted; partial ledger retained, no false completion")
                    attempt = task_root / f"attempt_{attempts:05d}"
                    if attempt.exists():
                        raise FileExistsError(f"Uncommitted attempt requires inspection, refusing overwrite: {attempt}")
                    seed_started = time.monotonic()
                    # Seed numbers may repeat across tasks/configs; identity is the full tuple.
                    seed = locked["jobs"][attempts]["seed"] if locked else opts.seed_start + attempts
                    row = dict(seed=seed, status="infrastructure_error", task=task, task_config=task_config)
                    attempts += 1
                    if server.poll() is not None:
                        raise RuntimeError("Policy server exited; refusing to consume/reject scene seeds")
                    if locked:
                        job = locked["jobs"][attempts-1]
                    else:
                        prepare = [str(opts.sim_python), str(ROOT / "scripts/internal/collect_robotwin_pool_worker.py"),
                                   "--prepare-seeds", "1", "--candidate-limit", "1", "--seed-start", str(seed),
                                   "--task-name", task, "--task-config", task_config, "--robotwin", str(opts.robotwin),
                                   "--output", str(attempt / "prepare"), "--port", str(server_opts.port),
                                   "--worker-id", str(lane), "--vector-env-checkout", str(ROOT / "third_party/RoboTwin_RLinf")]
                        runtime = attempt / "runtime"
                        runtime.mkdir(parents=True, mode=0o700, exist_ok=True)
                        sim_env = dict(env, CUDA_VISIBLE_DEVICES=str(sim_gpu), OIDN_DEFAULT_DEVICE="cuda",
                                       ROBONANA_SAPIEN_RENDER_DEVICE="cuda:0", XDG_RUNTIME_DIR=str(runtime))
                        rc = run_bounded(prepare, env=sim_env, log=attempt / "prepare.log", timeout=opts.seed_timeout)
                        if rc:
                            row.update(status="candidate_rejected", prepare_returncode=rc)
                            ledger.append(row)
                            atomic_json(ledger_path, ledger)
                            continue
                        job = json.loads((attempt / "prepare/accepted_seeds.json").read_text())["jobs"][0]
                    row["job"] = job
                    job["sampling_seed_base"] = int(seed) * 1000003
                    manifest = dict(task_name=task, task_config=task_config, jobs=[job], expert_validated=True)
                    atomic_json(attempt / "job.json", manifest)
                    command = [sys.executable, str(ROOT / "scripts/diagnostics/benchmark_robotwin_collection_pool.py"),
                               "--jobs-json", str(attempt / "job.json"), "--sim-gpus", str(sim_gpu),
                               "--server-gpu", str(opts.gpus[lane]), "--sim-python", str(opts.sim_python),
                               "--robotwin", str(opts.robotwin), "--checkpoint", str(opts.checkpoint),
                               "--model-config", str(opts.model_config), "--initial-dataset", str(opts.initial_dataset),
                               "--output", str(attempt / "rollout"), "--port", str(server_opts.port),
                               "--external-server", "--inference-mode", server_opts.inference_mode,
                               "--capture-mode", "scout_replay" if not locked else "scout",
                               "--candidate-batch-size", "32", "--timeout-seconds",
                               str(max(1, int(opts.seed_timeout - (time.monotonic()-seed_started))))]
                    rc = run_bounded(command, env=env, log=attempt / "rollout.log",
                                     timeout=max(1, opts.seed_timeout-(time.monotonic()-seed_started)) + 60)
                    row["returncode"] = rc
                    if rc == 0:
                        result = json.loads((attempt / "rollout/summary.json").read_text())
                        row.update(status="evaluated", result=result["episodes"][0])
                        accepted.append(job)
                        if not locked and not row["result"]["success"] and row["result"].get("replay_verified"):
                            # Publish an accepted-only dataset view. Rejected attempts and
                            # orphaned artifacts can never leak into Stage1 via a broad glob.
                            source = Path(row["result"]["hdf5"]).parent.parent.resolve()
                            link = opts.output / "failure_dataset" / f"{task_config}_{seed}" / task / "robonana_rollout"
                            link.parent.mkdir(parents=True, exist_ok=True)
                            if not link.exists():
                                link.symlink_to(source, target_is_directory=True)
                    ledger.append(row)
                    atomic_json(ledger_path, ledger)
                    if rc == 0 and result["replay_mismatches"]:
                        # Do not select scenes based on replay reproducibility/outcome.
                        raise RuntimeError("Scout/replay mismatch: preserve outcome, block failure dataset publication")
                if not locked:
                    atomic_json(task_root / "seeds.json", dict(
                        task_name=task, task_config=task_config, jobs=accepted, expert_validated=True,
                        robotwin_commit=revision, task_config_sha256=sha256_file(opts.robotwin / "task_config" / f"{task_config}.yml"),
                        sampling_seed_rule="seed * 1000003 + control_step // 48"))
                atomic_json(task_root / "summary.json", dict(
                    evaluated=len(accepted), errors=sum(r["status"] != "evaluated" for r in ledger),
                    successes=sum(bool(r["result"]["success"]) for r in ledger if r["status"] == "evaluated"),
                    paired_scene_count=len(locked["jobs"]) if locked else len(accepted)))
        finally:
            terminate_process_group(server, grace_seconds=10)


def collection(opts):
    import re
    tasks = re.findall(r"^([a-z0-9_]+):", (opts.robotwin / "task_config/_eval_step_limit.yml").read_text(), re.M)
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise ValueError("Expected exactly 50 unique upstream tasks")
    if opts.tasks:
        if not set(opts.tasks) <= set(tasks):
            raise ValueError("Unknown task in bounded probe")
        tasks = opts.tasks
    pairs = [(task, cfg) for task in tasks for cfg in ("demo_clean", "demo_randomized")]
    if len(opts.gpus) < 2 or len(opts.gpus) % 2 or len(set(opts.gpus)) != len(opts.gpus):
        raise ValueError("Use distinct GPU pairs; default 4 policy + 4 simulator GPUs")
    if opts.episodes <= 0 or opts.seed_timeout <= 0 or opts.seed_start < 0 or opts.candidate_multiplier <= 0:
        raise ValueError("Invalid episode count, seed or timeout")
    if opts.command == "eval" and not opts.manifests:
        raise ValueError("Paired evaluation requires locked collection --manifests")
    print(json.dumps(dict(pairs=pairs, episodes_per_config=opts.episodes, gpus=opts.gpus,
                          mode=opts.command, execute=opts.execute), indent=2))
    if not opts.execute:
        return
    # Frozen run fingerprint guards resume against accidentally switching the policy.
    from robonana.inference_contract import sha256_file
    signature = dict(checkpoint_sha256=sha256_file(opts.checkpoint), model_config_sha256=sha256_file(opts.model_config),
                     mode=opts.command, pairs=pairs, episodes=opts.episodes, seed_start=opts.seed_start,
                     manifests=str(opts.manifests.resolve()) if opts.manifests else None)
    signature = json.loads(json.dumps(signature))
    path = opts.output / "protocol.json"
    signature["robotwin_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=opts.robotwin, text=True).strip()
    signature["task_config_sha256"] = {cfg:sha256_file(opts.robotwin / "task_config" / f"{cfg}.yml")
                                      for cfg in ("demo_clean", "demo_randomized")}
    if path.exists() and json.loads(path.read_text()) != signature:
        raise ValueError("Output belongs to a different collection/eval protocol")
    atomic_json(path, signature)
    lanes = len(opts.gpus)//2
    with ThreadPoolExecutor(max_workers=lanes) as pool:
        futures = [pool.submit(collect_lane, opts, pairs[lane::lanes], lane) for lane in range(lanes)]
        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "aliases", "collect", "eval"))
    parser.add_argument("--phase", choices=("pretrain", "stage1", "stage2"), default="pretrain")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--replay-root", type=Path)
    gc = parser.add_mutually_exclusive_group()
    gc.add_argument("--no-gradient-checkpointing", action="store_true", help="Verified protocol default")
    gc.add_argument("--gradient-checkpointing", action="store_true", help="Explicit fallback to existing block checkpoint policy")
    parser.add_argument("--smoke-steps", type=int, default=0, help="Bounded 1..10 update probe; no checkpoints")
    parser.add_argument("--smoke-task-globs", help="Explicit certified-data subset for memory probes only")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--robotwin", type=Path, default=Path("/workspace/hongjia/RoboTwin"))
    parser.add_argument("--sim-python", type=Path, default=Path("/data3/hongjia/venvs/robotwin-sapien303/bin/python"))
    parser.add_argument("--initial-dataset", type=Path, default=Path("/workspace/datasets/fact-robotwin-v2/RoboTwin"))
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--tasks", nargs="+", help="Bounded smoke subset only; omit for full 50")
    parser.add_argument("--episodes", type=int, default=100, help="Per task AND per clean/random config")
    parser.add_argument("--seed-start", type=int, default=300000)
    parser.add_argument("--seed-timeout", type=int, default=1200)
    parser.add_argument("--candidate-multiplier", type=int, default=20, help="Maximum candidates per requested episode; lower for probes")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--manifests", type=Path)
    opts = parser.parse_args()
    sources = [str(ROOT / path) for path in
        ("src", "third_party/FACT", "third_party/flux2/src", "third_party/flux2_official/src")]
    sys.path[:0] = sources
    os.environ["PYTHONPATH"] = os.pathsep.join(sources) + os.pathsep + os.environ.get("PYTHONPATH", "")
    opts.output = opts.output.resolve()
    if opts.command in ("collect", "eval"):
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_args: STOP.set())
        if not opts.checkpoint or not opts.model_config:
            parser.error("Collection/eval requires checkpoint and model-config")
        collection(opts)
    else:
        training(opts)


if __name__ == "__main__":
    main()
