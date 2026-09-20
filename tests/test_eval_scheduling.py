"""统一调度的行为回归：多 worker 不重复领任务、重试不改变 seed。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from queue import Queue
from threading import Lock
import sys
import time
import pytest


def launcher():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_multitask_mbrl.py'
    spec = importlib.util.spec_from_file_location('eval_schedule', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_workers_share_one_model_and_claim_each_task_once(tmp_path, monkeypatch):
    m = launcher()
    monkeypatch.setitem(sys.modules, 'robotwin_eval_pool', SimpleNamespace(server_command=lambda *a: ['fake']))
    monkeypatch.setitem(sys.modules, 'eval_robotwin_task_isolated', SimpleNamespace(terminate_process_group=lambda *a, **kw: None))
    servers = []
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **kw: servers.append(a) or SimpleNamespace(poll=lambda:None))
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **kw: 'revision')
    queue = Queue()
    for i in range(12): queue.put((f'task{i}', 'demo_clean'))
    opts = SimpleNamespace(output=tmp_path, gpus=[0], port=9000, command='collect', robotwin=tmp_path,
                           workers_per_gpu=[2], task_queue=queue, shared_gpus=True)
    active, peak, seen = 0, 0, []
    lock = Lock()
    def task(opts, name, cfg, lane, server_opts, *rest):
        nonlocal active, peak
        assert server_opts.inference_batch_size == 1
        with lock:
            active += 1; peak = max(peak, active); seen.append(name)
        time.sleep(.02)
        with lock: active -= 1
    monkeypatch.setattr(m, 'collect_task', task)
    m.collect_lane(opts, [], 0)
    assert len(servers) == 1 and peak == 2
    assert len(set(seen)) == len(seen) == 12
    assert queue.unfinished_tasks == 0


def test_only_explicit_expert_rejection_consumes_seed(tmp_path, monkeypatch):
    m = launcher()
    command = ['worker', '--seed-start', '100000', '--output', 'placeholder']
    calls = []
    def fake(argv, **kw):
        calls.append(argv)
        output = Path(argv[-1])
        m.atomic_json(output/'rejected_seed.json', dict(seed=100000, reason='expert_infeasible'))
        return 1
    monkeypatch.setattr(m, 'run_bounded', fake)
    _, rejected = m.run_seed_stage(command, env={}, root=tmp_path, stage='prepare', timeout=1, retries=2)
    assert rejected and len(calls) == 1


def test_retry_exhaustion_retains_separate_artifacts(tmp_path, monkeypatch):
    m = launcher()
    seen = []
    monkeypatch.setattr(m, 'run_bounded', lambda argv, **kw: seen.append(list(argv)) or 124)
    with pytest.raises(RuntimeError, match='after 3 attempts'):
        m.run_seed_stage(['worker','--output','placeholder'], env={}, root=tmp_path,
                         stage='rollout', timeout=1, retries=2)
    assert len({argv[-1] for argv in seen}) == 3
    assert len(list(tmp_path.glob('*_status.json'))) == 3
