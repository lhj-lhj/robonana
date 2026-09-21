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
from queue import Queue, Empty

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [str(ROOT / name) for name in ('src','third_party/FACT','third_party/flux2_official/src')]
sys.path[:0] = SOURCES + [str(ROOT / 'scripts/internal')]
os.environ['PYTHONPATH'] = os.pathsep.join(SOURCES) + os.pathsep + os.environ.get('PYTHONPATH','')
STOP = threading.Event()


def atomic_json(path, value):
    # 对json进行原子写入
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_expert_jobs(root, task, task_config, episodes, *, allow_partial=False):
    """Consume a completed external harvest; never run expert checks here."""
    from robonana.sim.collection_pool import validate_jobs
    path = root / f"{task}__{task_config}" / "expert_manifest.json"
    manifest = json.loads(path.read_text())
    if (manifest.get("task_name"), manifest.get("task_config"), manifest.get("expert_validated")) != (task, task_config, True):
        raise ValueError(f"Expert manifest identity/validation mismatch: {path}")
    jobs = manifest["jobs"]
    validate_jobs(jobs, 1)
    if len(jobs) < episodes and not allow_partial:
        raise ValueError(f"Incomplete expert manifest: {path}: {len(jobs)}/{episodes}")
    return [dict(job) for job in jobs[:episodes]]


def run_bounded(command, *, env, log, timeout):
    """中文：给每个候选 seed 独立墙钟上限，清理自己启动的进程组。
    English: Reuse the isolated evaluator's process-group cleanup, including SIGTERM
    grace for the collector to close its separately-sessioned children.
    """
    from robonana.sim.processes import terminate_process_group
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


def run_seed_stage(command, *, env, root, stage, timeout, retries):
    """基础设施故障只重试同一命令/seed；每次单独保留日志和产物。"""
    for retry in range(retries + 1):
        output = root / (stage if retry == 0 else f"{stage}_retry_{retry}")
        argv = list(command)
        argv[argv.index("--output") + 1] = str(output)
        started = time.monotonic()
        rc = run_bounded(argv, env=env, log=root / f"{output.name}.log", timeout=timeout)
        rejected = output / "rejected_seed.json"
        if stage == "prepare" and rejected.exists():
            record = json.loads(rejected.read_text())
            seed = int(argv[argv.index("--seed-start") + 1])
            if record.get("seed") == seed and record.get("reason") in ("expert_infeasible", "expert_unstable"):
                return output, True
        atomic_json(root / f"{output.name}_status.json", dict(
            returncode=rc, elapsed_seconds=time.monotonic()-started, retry=retry))
        if rc == 0:
            return output, False
    # 不写入已消费 seed 的 ledger；重启前须检查该 attempt 的留档。
    raise RuntimeError(f"{stage} infrastructure failure after {retries+1} attempts: {root}")


def training(opts):
    from dataclasses import asdict
    from robonana.configs.training import build_training_config
    from robonana.configs.resume import build_resume_config
    config = build_resume_config(opts.options) if opts.command == 'resume' else build_training_config(opts.options)

    # 输出模型config信息
    batch = dict(gpus=len(config['launch']['gpu_ids']), microbatch=config['dataloaders']['train']['batch_size_per_gpu'],
                 accumulation_steps=config['train']['gradient_accumulation_steps'])
    batch['global_batch'] = batch['gpus'] * batch['microbatch'] * batch['accumulation_steps']
    # 用户看到的和实际传给 FACT 的是同一份 resolved config，不再二次 import 覆写。
    print(json.dumps(dict(requested=asdict(opts.options), batch=batch, resolved=config), indent=2, default=str))
    
    if not opts.execute and opts.command != 'audit':
        return
    if opts.command != 'resume':
        preflight_training(opts, config)
    if opts.command == 'audit':
        return
    
    # 防止覆写已有的项目目录
    project = Path(config['project_dir'])
    if project.exists():
        raise FileExistsError(f'Use a new output directory: {project}')
    project.mkdir(parents=True)
    # 把当前的执行experiment的config写入到project/'requested.json'
    atomic_json(project/'requested.json', json.loads(json.dumps(asdict(opts.options), default=str)))

    from fact_train import Config, launch_from_config
    # FACT 原生支持完整JSON；训练进程直接读取本次快照，不需要配置模块缓存。
    resolved = project/'launch_config.json'
    Config(config).save(str(resolved))
    launch_from_config(str(resolved))


def preflight_training(opts, config):
    from robonana.data.robotwin_hdf5 import RoboTwinHDF5Dataset, RoboTwinPosttrainSampler
    from robonana.data.robotwin_lerobot import RoboTwinLeRobotDataset
    from robonana.image_pipeline import validate_training_image_contracts, validate_episode_caches
    from torch.utils.data import ConcatDataset
    classes = {cls.__name__:cls for cls in (RoboTwinHDF5Dataset, RoboTwinLeRobotDataset)}
    data = config["dataloaders"]["train"]["data_or_config"]
    children = []
    try:
        for spec in data if isinstance(data, list) else [data]:
            child = classes[spec["_class_name"]].load(spec)
            children.append(child)
            child._ensure_index()
        if not opts.smoke_steps and ((opts.expected_original_episodes is not None and len(children[0].records) != opts.expected_original_episodes) or
                                    (opts.expected_tasks is not None and len({r.task_name for r in children[0].records}) != opts.expected_tasks)):
            raise ValueError(f"Dataset does not match explicit expected_original_episodes={opts.expected_original_episodes}, expected_tasks={opts.expected_tasks}")
        dataset = ConcatDataset(children) if len(children) > 1 else children[0]
        if len(children) > 1:
            RoboTwinPosttrainSampler(dataset, batch_size=opts.microbatch, pool_weights=
                config["dataloaders"]["train"]["sampler"]["pool_weights"])
        validate_training_image_contracts(dataset, config["models"]["checkpoint_dir"])
        summaries = [validate_episode_caches(child.records) for child in children]
        print("DATA PREFLIGHT PASSED: metadata, pools, A statistics, contracts AND all episode files", summaries)
    finally:
        for child in children:
            child.close()


def collect_task(opts, task, task_config, lane, server_opts, server, env, sim_gpu, revision):
    """一个 task/config 只有一个调度者，避免并发重复 seed 或覆盖 ledger。"""
    from robonana.inference_contract import sha256_file
    task_root = opts.output / task / task_config
    expert_jobs = getattr(opts, "expert_jobs", None)
    locked = None
    if expert_jobs is not None:
        shard = opts.shard_offset + lane
        jobs = expert_jobs[f"{task}__{task_config}"][shard::opts.shard_count]
        if not jobs:
            return
        locked = dict(jobs=jobs)
        task_root = task_root / f"shard_{shard:02d}"
    task_root.mkdir(parents=True, exist_ok=True)
    ledger_path = task_root / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    if any(row.get("result", {}).get("replay_verified") is False for row in ledger):
        print(f"Warning: {task_root}: retaining scout SR; unverified replays remain excluded", flush=True)
    accepted = [r["job"] for r in ledger if r["status"] == "evaluated"]
    if opts.manifests and expert_jobs is None:
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
            if not opts.resume_interrupted:
                raise FileExistsError(f"Uncommitted attempt requires inspection, refusing overwrite: {attempt}")
            # 持有全局运行锁后才允许归档；不删除、不把半成品记为失败、不跳 seed。
            archive = task_root / 'interrupted' / f'{attempt.name}_{time.time_ns()}'
            archive.parent.mkdir(exist_ok=True)
            attempt.rename(archive)
        # Seed numbers may repeat across tasks/configs; identity is the full tuple.
        seed = locked["jobs"][attempts]["seed"] if locked else opts.seed_start + attempts
        row = dict(seed=seed, status="infrastructure_error", task=task, task_config=task_config)
        attempts += 1
        if server.poll() is not None:
            raise RuntimeError("Policy server exited; refusing to consume/reject scene seeds")
        if locked:
            job = dict(locked["jobs"][attempts-1])
        else:
            prepare = [str(opts.sim_python), str(ROOT / "scripts/internal/collect_robotwin_pool_worker.py"),
                       "--prepare-seeds", "1", "--strict-infra", "--candidate-limit", "1", "--seed-start", str(seed),
                       "--task-name", task, "--task-config", task_config, "--robotwin", str(opts.robotwin),
                       "--output", str(attempt / "prepare"), "--port", str(server_opts.port),
                       "--worker-id", str(lane)]
            runtime = attempt / "runtime"
            runtime.mkdir(parents=True, mode=0o700, exist_ok=True)
            sim_env = dict(env, CUDA_VISIBLE_DEVICES=str(sim_gpu), OIDN_DEFAULT_DEVICE="cuda",
                           ROBONANA_SAPIEN_RENDER_DEVICE="cuda:0", XDG_RUNTIME_DIR=str(runtime))
            prepared, rejected = run_seed_stage(prepare, env=sim_env, root=attempt,
                stage="prepare", timeout=opts.seed_timeout, retries=opts.infra_retries)
            if rejected:
                row.update(status="candidate_rejected", reason=json.loads((prepared / "rejected_seed.json").read_text())["reason"])
                ledger.append(row)
                atomic_json(ledger_path, ledger)
                continue
            job = json.loads((prepared / "accepted_seeds.json").read_text())["jobs"][0]
        row["job"] = job
        job["sampling_seed_base"] = int(seed) * 1000003
        manifest = dict(task_name=task, task_config=task_config, jobs=[job], expert_validated=True)
        atomic_json(attempt / "job.json", manifest)
        command = [sys.executable, str(ROOT / "scripts/internal/robotwin_eval_pool.py"),
                   "--jobs-json", str(attempt / "job.json"), "--sim-gpus", str(sim_gpu),
                   "--server-gpu", str(opts.gpus[lane]), "--sim-python", str(opts.sim_python),
                   "--robotwin", str(opts.robotwin), "--checkpoint", str(opts.checkpoint),
                   "--model-config", str(opts.model_config), "--initial-dataset", str(opts.initial_dataset),
                   "--flux-checkpoint-dir", str(opts.flux_checkpoint_dir), "--stats-path", str(opts.stats_path),
                   "--output", str(attempt / "rollout"), "--port", str(server_opts.port),
                   "--external-server", "--inference-mode", server_opts.inference_mode,
                   "--capture-mode", opts.capture_mode,
                   "--candidate-batch-size", str(opts.candidate_batch_size), "--timeout-seconds",
                   str(opts.seed_timeout)]
        rollout, _ = run_seed_stage(command, env=env, root=attempt, stage="rollout",
            timeout=opts.seed_timeout + 60, retries=opts.infra_retries)
        row["returncode"] = 0
        result = json.loads((rollout / "summary.json").read_text())
        if len(result['episodes']) != 1 or int(result['episodes'][0]['seed']) != seed:
            raise ValueError(f"Rollout result does not match assigned seed {seed}: {rollout}")
        row.update(status="evaluated", result=result["episodes"][0])
        accepted.append(job)
        if opts.capture_mode == "scout_replay" and not row["result"]["success"] and row["result"].get("replay_verified"):
            # Publish an accepted-only dataset view. Rejected attempts and
            # orphaned artifacts can never leak into Stage1 via a broad glob.
            source = Path(row["result"]["hdf5"]).parent.parent.resolve()
            link = opts.output / "failure_dataset" / f"{task_config}_{seed}" / task / "robonana_rollout"
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(source, target_is_directory=True)
        ledger.append(row)
        atomic_json(ledger_path, ledger)
        # A replay mismatch invalidates only that failure artifact.  The
        # evaluated scout outcome stays in the ledger, while the
        # replay_verified guard above keeps the artifact out of the
        # published failure dataset.  Continue the remaining jobs.
    if not locked or expert_jobs is not None:
        atomic_json(task_root / "seeds.json", dict(
            task_name=task, task_config=task_config, jobs=accepted, expert_validated=True,
            robotwin_commit=revision, task_config_sha256=sha256_file(opts.robotwin / "task_config" / f"{task_config}.yml"),
            sampling_seed_rule="seed * 1000003 + control_step // 48"))
    (task_root / "blocked.json").unlink(missing_ok=True)
    atomic_json(task_root / "summary.json", dict(
        evaluated=len(accepted), errors=sum(r["status"] != "evaluated" for r in ledger),
        successes=sum(bool(r["result"]["success"]) for r in ledger if r["status"] == "evaluated"),
        paired_scene_count=len(locked["jobs"]) if locked else len(accepted)))


def export_dataset(opts, pairs):
    """旧训练采集只需要平铺视图：硬链接已验收文件，不再执行第二套 eval。"""
    for task, config in pairs:
        ledger = json.loads((opts.output / task / config / "ledger.json").read_text())
        for row in ledger:
            if row['status'] != 'evaluated':
                continue
            source = Path(row['result']['hdf5'])
            target = opts.export_dataset / task / 'robonana_rollout/data' / f"episode{row['seed']}.hdf5"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not source.samefile(target):
                    raise FileExistsError(f"Refusing to overwrite unrelated dataset artifact: {target}")
            else:
                os.link(source, target)


def collect_lane(opts, pairs, lane):
    from robotwin_eval_pool import server_command
    from robonana.sim.processes import terminate_process_group
    from robonana.inference_contract import sha256_file
    server_opts = copy(opts)
    server_opts.port = opts.port + lane
    server_opts.inference_mode = opts.inference_mode
    server_opts.inference_batch_size, server_opts.batch_wait_ms = 1, 0
    lane_root = opts.output / f"lane_{lane}"
    lane_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(opts.gpus[lane]),
               ROBONANA_REJECTION_CANDIDATE_BATCH_SIZE=str(opts.candidate_batch_size), FACT_ROBOTWIN_EVAL_VIDEO_LOG="0")
    sim_gpu = opts.gpus[lane] if opts.shared_gpus else opts.gpus[lane + len(opts.gpus)//2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=opts.robotwin, text=True).strip()
    # 遥测组件拒绝覆盖文件；每次重启归档旧遥测，保留历史而不阻断模型启动。
    metrics = lane_root / 'batch_metrics.jsonl'
    if metrics.exists():
        metrics.rename(lane_root / f'batch_metrics_{time.time_ns()}.jsonl')
    with (lane_root / "server.log").open("a") as log:
        server = subprocess.Popen(server_command(server_opts, lane_root), cwd=ROOT, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            # 一份模型服务供多个独立仿真进程使用；SAPIEN 不共享进程内全局 RNG。
            work = getattr(opts, "task_queue", None)
            if work is None:
                work = Queue()
                for pair in pairs:
                    work.put(pair)
            failures = []
            def consume():
                while not STOP.is_set():
                    try:
                        task, task_config = work.get_nowait()
                    except Empty:
                        return
                    try:
                        collect_task(opts, task, task_config, lane, server_opts, server,
                                     env, sim_gpu, revision)
                    except Exception as exc:
                        # 单个场景的重试耗尽不拖停其他配置；它保留为待处理，不计入 SR。
                        failures.append(exc)
                        atomic_json(opts.output / task / task_config / 'blocked.json',
                                    dict(error=repr(exc), lane=lane, time=time.time()))
                        if server.poll() is not None or STOP.is_set():
                            return
                    finally:
                        work.task_done()
            counts = opts.workers_per_gpu
            count = counts[0] if len(counts) == 1 else counts[lane]
            with ThreadPoolExecutor(max_workers=count) as workers:
                pending = [workers.submit(consume) for _ in range(count)]
                for future in pending:
                    future.result()
            if failures:
                raise RuntimeError(f"Lane {lane}: {len(failures)} blocked configs; inspect blocked.json") from failures[0]
        finally:
            terminate_process_group(server, grace_seconds=10)


def write_results(opts, pairs):
    """从同一份 ledger 导出旧比较脚本使用的 CSV；expert 拒绝不进入 SR 分母。"""
    import csv
    target = opts.output / 'results.csv'
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['task', 'task_config', 'success', 'total', 'success_rate'])
        writer.writeheader()
        for task, config in pairs:
            root = opts.output / task / config
            paths = [root / 'ledger.json'] if (root / 'ledger.json').exists() else sorted(root.glob('shard_*/ledger.json'))
            rows = [row for path in paths for row in json.loads(path.read_text()) if row['status'] == 'evaluated']
            successes = sum(bool(row['result']['success']) for row in rows)
            writer.writerow(dict(task=task, task_config=config, success=successes, total=len(rows),
                                 success_rate=successes/len(rows) if rows else 'ERROR'))
    temporary.replace(target)


def collection(opts):
    if not opts.execute:
        return _collection(opts)
    # 两个 supervisor 不能同时写同一套 ledger；重启前旧进程必须已退出。
    import fcntl
    opts.output.mkdir(parents=True, exist_ok=True)
    with (opts.output / '.eval.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another evaluator owns this output directory') from exc
        return _collection(opts)


def _collection(opts):
    import re
    tasks = re.findall(r"^([a-z0-9_]+):", (opts.robotwin / "task_config/_eval_step_limit.yml").read_text(), re.M)
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise ValueError("Expected exactly 50 unique upstream tasks")
    if opts.tasks:
        if not set(opts.tasks) <= set(tasks):
            raise ValueError("Unknown task in bounded probe")
        tasks = opts.tasks
    pairs = [(task, cfg) for task in tasks for cfg in opts.task_configs]
    if opts.export_dataset and (len(opts.task_configs) != 1 or opts.capture_mode != "full"):
        raise ValueError("Dataset export requires full capture and a single task config")
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate task/config pairs")
    if opts.infra_retries < 0:
        raise ValueError("infra-retries must be nonnegative")
    if len(set(opts.gpus)) != len(opts.gpus) or not opts.gpus or any(g < 0 for g in opts.gpus):
        raise ValueError("Use distinct nonnegative GPU ids")
    if not opts.shared_gpus and (len(opts.gpus) < 2 or len(opts.gpus) % 2):
        raise ValueError("Use distinct GPU pairs; default 4 policy + 4 simulator GPUs")
    if opts.episodes <= 0 or opts.seed_timeout <= 0 or opts.seed_start < 0 or opts.candidate_multiplier <= 0:
        raise ValueError("Invalid episode count, seed or timeout")
    
    lanes = len(opts.gpus) if opts.shared_gpus else len(opts.gpus)//2
    # A shared lane can either consume frozen expert jobs or run the official
    # expert check inline.  The latter preserves RoboTwin's canonical candidate
    # sequence while still colocating one persistent policy server and simulator
    # on every GPU.
    counts = opts.workers_per_gpu
    if len(counts) not in (1, lanes) or any(n < 1 or n > 4 for n in counts):
        raise ValueError("workers-per-gpu needs one count or a count per lane, each 1..4")
    if opts.expert_seed_cache and any(n != 1 for n in counts):
        raise ValueError("Frozen cross-host shards currently require one worker per lane")
    if opts.expert_seed_cache and opts.manifests:
        raise ValueError("Choose external expert cache or locked collection manifests")
    opts.expert_jobs = None
    if opts.expert_seed_cache:
        if opts.allow_partial_expert_seeds and not opts.tasks:
            raise ValueError("Partial expert manifests require an explicit bounded task subset")
        if opts.shard_offset < 0 or opts.shard_count < opts.shard_offset + lanes:
            raise ValueError("Shard range must include every local lane")
        opts.expert_jobs = {}
        ready = []
        for task, cfg in pairs:
            path = opts.expert_seed_cache / f"{task}__{cfg}" / "expert_manifest.json"
            if opts.ready_only and (not path.exists() or len(json.loads(path.read_text()).get("jobs", [])) < opts.episodes):
                print(f"Pending expert cache: {task}/{cfg}")
                continue
            opts.expert_jobs[f"{task}__{cfg}"] = load_expert_jobs(
                opts.expert_seed_cache, task, cfg, opts.episodes, allow_partial=opts.allow_partial_expert_seeds)
            ready.append((task, cfg))
        pairs = ready
        if not pairs:
            raise ValueError("No completed expert manifests are ready")
    print(json.dumps(dict(pairs=pairs, episodes_per_config=opts.episodes, gpus=opts.gpus,
                          mode=opts.command, execute=opts.execute), indent=2))
    if not opts.execute:
        return
    # Frozen run fingerprint guards resume against accidentally switching the policy.
    from robonana.inference_contract import sha256_file
    signature = dict(checkpoint_sha256=sha256_file(opts.checkpoint), model_config_sha256=sha256_file(opts.model_config),
                     mode=opts.command, pairs=pairs, episodes=opts.episodes, seed_start=opts.seed_start,
                     manifests=str(opts.manifests.resolve()) if opts.manifests else None)
    signature.update(inference_mode=opts.inference_mode, capture_mode=opts.capture_mode)
    signature = json.loads(json.dumps(signature))
    if opts.expert_jobs is not None:
        signature.update(expert_jobs=opts.expert_jobs, shard_count=opts.shard_count,
                         shard_offset=opts.shard_offset, lanes=lanes, shared_gpus=opts.shared_gpus)
    path = opts.output / "protocol.json"
    signature["robotwin_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=opts.robotwin, text=True).strip()
    signature["task_config_sha256"] = {cfg:sha256_file(opts.robotwin / "task_config" / f"{cfg}.yml")
                                      for cfg in ("demo_clean", "demo_randomized")}
    if path.exists() and json.loads(path.read_text()) != signature:
        raise ValueError("Output belongs to a different collection/eval protocol")
    atomic_json(path, signature)
    from dataclasses import asdict
    # 用户输入与实际派生的拓扑完整留档，调整并发不能偷偷改变冻结的模型/seed协议。
    atomic_json(opts.output / 'eval_config.json', dict(
        requested=json.loads(json.dumps(asdict(opts.options), default=str)),
        resolved=dict(pairs=pairs, policy_lanes=lanes, inference_batch_size=1,
                      policy_ports=list(range(opts.port, opts.port+lanes)), workers_per_lane=counts)))
    # 运行参数独立记录：调整并发不能改变冻结的 seed/模型协议。
    atomic_json(opts.output / "execution.json", dict(workers_per_gpu=counts, gpus=opts.gpus,
                inference_batch_size=1, updated_at=time.time()))
    if opts.expert_jobs is None:
        opts.task_queue = Queue()
        for pair in pairs:
            opts.task_queue.put(pair)
    with ThreadPoolExecutor(max_workers=lanes) as pool:
        futures = [pool.submit(collect_lane, opts, pairs if opts.expert_jobs is not None else pairs[lane::lanes], lane)
                   for lane in range(lanes)]
        for future in futures:
            future.result()
    write_results(opts, pairs)
    if opts.export_dataset:
        export_dataset(opts, pairs)


def main():
    from dataclasses import asdict
    from robonana.configs.schema import load_options
    from robonana.configs.training import TrainOptions
    from robonana.configs.evaluation import EvalOptions
    from robonana.configs.resume import ResumeOptions
    parser = argparse.ArgumentParser(description="One explicit JSON configuration; no legacy environment overrides")
    parser.add_argument('command', choices=('train','resume','audit','eval'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Dry-run unless explicitly requested')
    args = parser.parse_args()
    # legacy = sorted(key for key in os.environ if key.startswith('ROBONANA_'))
    # if legacy:
    #     parser.error(f'Legacy experiment environment overrides are unsupported; unset {legacy} and use --config')
    cls = EvalOptions if args.command=='eval' else ResumeOptions if args.command=='resume' else TrainOptions
    try:
        options = load_options(cls, args.config)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    # Show the anchor as well as resolved paths; changing shell CWD must not change assets.
    print(json.dumps(dict(config_file=str(args.config.resolve()), path_base=str(args.config.resolve().parent))), file=sys.stderr)
    opts = argparse.Namespace(**asdict(options), command=args.command, execute=args.execute, options=options)
    if args.command=='eval':
        print(json.dumps(dict(requested=asdict(options), inference_batch_size=1), indent=2, default=str))
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_args: STOP.set())
        collection(opts)
    else:
        training(opts)


if __name__ == '__main__':
    main()
