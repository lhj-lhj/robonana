"""中文：固定协议回归。 English: budget, dataset, optimizer and entrypoint regressions."""
import importlib
import os

import pytest


def test_original_flux_loading(tmp_path):
    import torch
    from safetensors.torch import save_file
    from flux2.model import Flux2
    from test_pretrained import tiny_params
    from robonana.models.pretrained import load_flux2_backbone_checkpoint
    params = tiny_params()
    original = Flux2(params)
    path = tmp_path / "flux.safetensors"
    save_file(original.state_dict(),str(path))
    model, report = load_flux2_backbone_checkpoint(path, params=params, action_dim=6,
        state_dim=5, expert_hidden_dim=32, dtype=torch.float32)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    assert "value_expert.query.weight" in report.initialized_robot_parameters
    bad = dict(original.state_dict())
    bad.pop(next(iter(bad)))
    save_file(bad,str(path))
    with pytest.raises(ValueError, match="keys mismatch"):
        load_flux2_backbone_checkpoint(path,params=params,expert_hidden_dim=32)


def test_bounded_process_timeout(tmp_path):
    import importlib.util
    from pathlib import Path
    import sys
    import time
    if os.name == "nt":
        pytest.skip("POSIX process group watchdog validated on 190")
    path=Path(__file__).resolve().parents[1]/"scripts/run_multitask_mbrl.py"
    spec=importlib.util.spec_from_file_location("protocol_launcher",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    start=time.monotonic()
    rc=module.run_bounded([sys.executable,"-c","import time; time.sleep(90)"],
                           env=dict(os.environ),log=tmp_path/"sleep.log",timeout=.2)
    assert rc==124 and time.monotonic()-start<5


def test_seed_timeout_retries_same_seed_and_locked_eval_stops(tmp_path, monkeypatch):
    """Run the real lane supervisor with fake simulator processes, not GPU jobs."""
    import importlib.util
    import json
    from pathlib import Path
    import sys
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[1]/"scripts/run_multitask_mbrl.py"
    spec=importlib.util.spec_from_file_location("protocol_lane_test",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules,"robotwin_eval_pool",
                        SimpleNamespace(server_command=lambda *args:["fake-server"]))
    monkeypatch.setitem(sys.modules,"robonana.sim.processes",
                        SimpleNamespace(terminate_process_group=lambda *args,**kw:None))
    monkeypatch.setattr(module.subprocess,"Popen",lambda *args,**kw:SimpleNamespace(poll=lambda:None))
    monkeypatch.setattr(module.subprocess,"check_output",lambda *args,**kw:"revision\n")
    simulator=tmp_path/"sim"
    (simulator/"task_config").mkdir(parents=True)
    (simulator/"task_config/demo_clean.yml").write_text("config")
    opts=SimpleNamespace(command="collect",output=tmp_path/"collect",robotwin=simulator,
        checkpoint=tmp_path/"model.bin",model_config=tmp_path/"model.json",gpus=[0,1],port=8400,
        episodes=1,seed_timeout=60,seed_start=300000,candidate_multiplier=20,sim_python=Path(sys.executable),
        initial_dataset=tmp_path/"initial",manifests=None, shared_gpus=False, workers_per_gpu=[1],
        infra_retries=2, resume_interrupted=False, capture_mode="scout_replay", inference_mode="action_only",
        candidate_batch_size=32, flux_checkpoint_dir=tmp_path, stats_path=tmp_path/"stats.json")
    prepared=[]
    def fake_run(command,**kwargs):
        if "--prepare-seeds" in command:
            seed=int(command[command.index("--seed-start")+1])
            prepared.append(seed)
            if len(prepared)==1:
                return 124
            output=Path(command[command.index("--output")+1])
            module.atomic_json(output/"accepted_seeds.json",dict(jobs=[dict(seed=seed,instruction="fixed words")]))
        else:
            output=Path(command[command.index("--output")+1])
            module.atomic_json(output/"summary.json",dict(replay_mismatches=0,
                episodes=[dict(seed=300000,success=True,hdf5=None)]))
        return 0
    monkeypatch.setattr(module,"run_bounded",fake_run)
    module.collect_lane(opts,[("hanging_mug","demo_clean")],0)
    locked_path=opts.output/"hanging_mug/demo_clean/seeds.json"
    locked=locked_path.read_bytes()
    assert prepared==[300000,300000]
    assert json.loads(locked)["jobs"][0]["seed"]==300000
    opts.command="eval"
    opts.capture_mode="scout"
    opts.manifests=opts.output
    opts.output=tmp_path/"eval"
    eval_commands=[]
    def error_run(command,**kwargs):
        eval_commands.append(command)
        assert "--prepare-seeds" not in command
        assert command[command.index("--capture-mode")+1]=="scout"
        return 124
    monkeypatch.setattr(module,"run_bounded",error_run)
    with pytest.raises(RuntimeError, match="blocked configs"):
        module.collect_lane(opts,[("hanging_mug","demo_clean")],0)
    assert len(eval_commands)==3 and locked_path.read_bytes()==locked
    assert not (opts.output/"hanging_mug/demo_clean/ledger.json").exists()
    assert not (opts.output/"hanging_mug/demo_clean/summary.json").exists()


def test_expert_cache_shared_lanes_skip_prepare_and_keep_failures(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    import sys
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / "scripts/run_multitask_mbrl.py"
    spec = importlib.util.spec_from_file_location("expert_cache_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "robotwin_eval_pool", SimpleNamespace(server_command=lambda *a: ["server"]))
    monkeypatch.setitem(sys.modules, "robonana.sim.processes", SimpleNamespace(terminate_process_group=lambda *a, **k: None))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "revision\n")
    task, cfg = "place_dual_shoes", "demo_clean"
    jobs = [dict(seed=s, instruction=f"instruction {s}") for s in range(100)]
    cache = tmp_path / "cache"
    manifest = cache / f"{task}__{cfg}" / "expert_manifest.json"
    module.atomic_json(manifest, dict(task_name=task, task_config=cfg, expert_validated=True, jobs=jobs))
    loaded = module.load_expert_jobs(cache, task, cfg, 100)
    seen = []
    opts = SimpleNamespace(command="collect", output=tmp_path/"run", robotwin=tmp_path/"robotwin",
        checkpoint=tmp_path/"model.bin", model_config=tmp_path/"config.json", gpus=list(range(8)), port=8400,
        episodes=100, seed_timeout=60, seed_start=300000, candidate_multiplier=20, sim_python=Path(sys.executable),
        initial_dataset=tmp_path/"initial", manifests=None, expert_jobs={f"{task}__{cfg}": loaded},
        shared_gpus=True, shard_count=8, shard_offset=0, workers_per_gpu=[1], infra_retries=2, resume_interrupted=False, capture_mode="scout_replay", inference_mode="action_only",
        candidate_batch_size=32, flux_checkpoint_dir=tmp_path, stats_path=tmp_path/"stats.json")
    (opts.robotwin / "task_config").mkdir(parents=True)
    (opts.robotwin / f"task_config/{cfg}.yml").write_text("config")
    def rollout(command, **kwargs):
        assert "--prepare-seeds" not in command
        assert command[command.index("--capture-mode")+1] == "scout_replay"
        assert command[command.index("--sim-gpus")+1] == command[command.index("--server-gpu")+1]
        job = json.loads(Path(command[command.index("--jobs-json")+1]).read_text())["jobs"][0]
        seen.append(job["seed"])
        assert job["sampling_seed_base"] == job["seed"] * 1000003
        output = Path(command[command.index("--output")+1])
        hdf5 = output/"dataset"/task/"robonana_rollout/data/episode0.hdf5"
        hdf5.parent.mkdir(parents=True)
        hdf5.touch()
        mismatch = job["seed"] == 0
        module.atomic_json(output/"summary.json", dict(replay_mismatches=int(mismatch), episodes=[dict(
            seed=job["seed"], success=False, replay_verified=not mismatch, hdf5=str(hdf5))]))
        return 0
    monkeypatch.setattr(module, "run_bounded", rollout)
    for lane in range(8):
        module.collect_lane(opts, [(task,cfg)], lane)
    assert sorted(seen) == list(range(100))
    assert len(list((opts.output/"failure_dataset").glob(f"*/{task}/robonana_rollout"))) == 99
    first_ledger = json.loads((opts.output/task/cfg/"shard_00/ledger.json").read_text())
    assert first_ledger[0]["status"] == "evaluated"
    assert first_ledger[0]["result"]["replay_verified"] is False
    assert jobs == loaded  # Scheduling never mutates the frozen input manifests.
    with pytest.raises(ValueError, match="Incomplete"):
        module.load_expert_jobs(cache, task, cfg, 101)
    assert len(module.load_expert_jobs(cache, task, cfg, 101, allow_partial=True)) == 100
    duplicate = dict(task_name=task, task_config=cfg, expert_validated=True, jobs=[jobs[0],jobs[0]])
    module.atomic_json(manifest, duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        module.load_expert_jobs(cache, task, cfg, 2)


