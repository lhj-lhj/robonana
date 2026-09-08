#!/usr/bin/env python3
# 中文：内部辅助：运行单个持久化采集环境与客户端。
# English: Internal helper: run one persistent collection environment and client.
# 调用 / Invocation: 由并行采集管理器启动，不作为用户独立入口。 / Spawned by the collection supervisor, not a standalone user entry.
# 导航 / Guide: scripts/README.md (internal)
"""One GPU-bound persistent environment/client; launched by the pool supervisor."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

from robonana.sim import configure_sapien_runtime
from robonana.sim.collection_pool import EpisodeQueue, RoboNanaSubEnv, load_vector_env, make_vector_env
# 中文：复用仿真环境适配。 English: reuse the simulator environment bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env"))
from robotwin_eval_bootstrap import _install_static_camera_filter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--seed-start", type=int, default=100100)
    parser.add_argument("--task-name", default='hanging_mug')
    parser.add_argument("--task-config", default='demo_clean')
    parser.add_argument("--robotwin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--vector-env-checkout", type=Path, required=True)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--prepare-seeds", type=int, default=0,
                        help="Expert-check this many new seeds, without loading a policy or recording replay")
    opts = parser.parse_args()
    if not opts.jobs and not opts.prepare_seeds:
        parser.error('--jobs is required for collection')
    payload = json.loads(opts.jobs.read_text()) if opts.jobs else dict(
        task_name=opts.task_name, task_config=opts.task_config, jobs=[dict(seed=opts.seed_start)])
    output = opts.output.resolve()
    vector_checkout = opts.vector_env_checkout.resolve()
    if not opts.prepare_seeds and opts.queue is None:
        parser.error('--queue is required for collection')
    queue = EpisodeQueue(opts.queue.resolve()) if opts.queue else None
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(opts.robotwin.resolve())
    sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "script")]
    configure_sapien_runtime()
    # Reuse the production adapter's Warp compatibility, stable diffusion seed,
    # model client and per-control-step recording. No new inference implementation.
    import robonana_robotwin_client as adapter
    namespace = runpy.run_path("script/eval_policy.py", run_name="_collection_pool")
    _install_static_camera_filter()
    vector_class = load_vector_env(vector_checkout)

    def collect(_name, task, args, model, _seed, **kwargs):
        args.update(eval_mode=True, render_freq=0, eval_video_log=False)
        args.pop("eval_video_save_dir", None)
        if opts.prepare_seeds:
            # 中文：沿用官方 eval_policy.py 的 expert-check 与语言生成顺序。
            # English: same expert acceptance and post-reset instruction generation as
            # https://github.com/RoboTwin-Platform/RoboTwin/blob/main/script/eval_policy.py
            # No policy-success filtering: expert feasibility only. Save actual seeds,
            # not a range that might silently change after rejection in a later loop.
            import numpy as np
            accepted = []
            start_seed = int(payload['jobs'][0]['seed'])
            for seed in range(start_seed, start_seed + opts.prepare_seeds * 20):
                try:
                    task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
                    episode = task.play_once()
                    valid = bool(task.plan_success and task.check_success())
                finally:
                    task.close_env()
                if not valid:
                    continue
                try:
                    task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
                    descriptions = namespace['generate_episode_descriptions'](
                        args['task_name'], [episode['info']], opts.prepare_seeds)
                    instruction = str(np.random.choice(descriptions[0]['seen']))
                finally:
                    task.close_env()
                accepted.append(dict(seed=seed, instruction=instruction,
                                     source='official_expert_check'))
                manifest = dict(task_name=payload['task_name'], task_config=payload['task_config'],
                                jobs=accepted, expert_validated=True)
                temporary = output / 'accepted_seeds.json.tmp'
                temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
                temporary.replace(output / 'accepted_seeds.json')
                print(json.dumps({'accepted': len(accepted), 'seed': seed}), flush=True)
                if len(accepted) == opts.prepare_seeds:
                    return seed + 1, 0
            raise RuntimeError('Seed preparation exhausted its bounded candidate range')
        task.suc = 0
        task.test_num = 0
        results = []
        slot = RoboNanaSubEnv(task, model, payload["jobs"], args, adapter)
        vector = make_vector_env(slot, vector_class)
        try:
            while (job := queue.claim(opts.worker_id)) is not None:
                start = time.perf_counter()
                vector.reset(env_idx=[0], env_seeds=[int(job["seed"])])
                while not slot.done:
                    _, _, _, _, infos = vector.step([None])
                    if infos[0]["steps"] % 48 == 0:
                        print(json.dumps({"progress": infos[0], "worker": opts.worker_id}), flush=True)
                result = infos[0]
                result["duration_seconds"] = time.perf_counter() - start
                results.append(result)
                with (output / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                print(json.dumps(result), flush=True)
                queue.complete(job["seed"], opts.worker_id, result)
        finally:
            vector.close()
            vector.env_thread_pool.shutdown(wait=True)
        (output / "complete.json").write_text(json.dumps(results), encoding="utf-8")
        return int(payload["jobs"][-1]["seed"]) + 1, sum(r["success"] for r in results)

    main_fn = namespace["main"]
    main_fn.__globals__["eval_function_decorator"] = lambda _policy, name: (
        (lambda _args: None) if opts.prepare_seeds and name == 'get_model' else getattr(adapter, name))
    main_fn.__globals__["eval_policy"] = collect
    main_fn({"task_name": payload["task_name"], "task_config": payload["task_config"],
             "policy_name": "robonana_robotwin.adapter", "ckpt_setting": "pool_probe",
             "instruction_type": "seen", "seed": 0, "test_num": len(payload["jobs"]),
             "host": "127.0.0.1", "port": opts.port, "action_dim": 14,
             "execute_actions_per_plan": 48, "server_wait_seconds": 600,
             "server_timeout_ms": 600000, "low_frequency_rgb": False,
             "skip_action_render_sync": False, "enable_value_vis": False,
             "trace_value_only": True})


if __name__ == "__main__":
    main()
