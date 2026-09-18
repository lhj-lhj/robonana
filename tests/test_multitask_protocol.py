"""中文：固定协议回归。 English: budget, dataset, optimizer and entrypoint regressions."""
import importlib
import os

import pytest


@pytest.mark.parametrize("phase,steps,rates,keeps", [
    ("pretrain",120000,(2e-5,1e-4),[10000,30000,60000,120000]),
    ("stage1",60000,(2e-5,2e-5),[10000,30000,60000]),
    ("stage2",20000,(1e-4,1e-4),[10000,20000]),
])
def test_protocol_config(monkeypatch, phase, steps, rates, keeps):
    monkeypatch.setenv("ROBONANA_PROTOCOL_PHASE", "pretrain")
    monkeypatch.setenv("ROBONANA_MAC_PRETRAIN_CHECKPOINT", "/explicit/source.bin")
    monkeypatch.setenv("ROBONANA_MAC_PRETRAIN_CONFIG", "/explicit/config.json")
    monkeypatch.setenv("ROBONANA_REPLAY_ROOT", "/explicit/failures")
    from robonana.configs.robotwin_flux2 import config as base
    from robonana.configs.multitask_mbrl import build_protocol_config
    config = build_protocol_config(base, phase)
    train, loader = config["train"], config["dataloaders"]["train"]
    assert config["launch"]["gpu_ids"] == list(range(8))
    assert loader["batch_size_per_gpu"] * 8 * train["gradient_accumulation_steps"] == 128
    assert train["max_steps"] == config["schedulers"]["decay_steps"] == steps
    assert (config["optimizers"]["lr"],config["optimizers"]["robot_lr"]) == rates
    assert train["checkpoint_keeps"] == keeps
    assert train["checkpoint_save_optimizer"] and not train["resume"]
    assert train["mixed_precision"] == "bf16"
    assert train["posttrain"]["ema"]["target"] == "value_expert_only"
    if phase == "pretrain":
        assert loader["data_or_config"]["task_globs"] == ("Clean/*", "Randomized/*")
        assert config["models"]["initialization"] == "flux_backbone"
    else:
        from robonana.data.robotwin_hdf5 import RoboTwinHDF5Dataset
        from robonana.data.robotwin_lerobot import RoboTwinLeRobotDataset
        classes = {c.__name__:c for c in (RoboTwinHDF5Dataset,RoboTwinLeRobotDataset)}
        for pool in loader["data_or_config"]:
            classes[pool["_class_name"]].load(pool).close()
        assert loader["sampler"]["pool_weights"] == dict(original_success=.5, collected_success_replay=0.,
                                                          historical_failure_replay=0., latest_failure=.5)
    assert base["dataloaders"]["train"]["data_or_config"]["task_globs"] == ("Clean/hanging_mug",)


def test_bounded_checkpoint_probe(monkeypatch):
    from robonana.configs.multitask_mbrl import build_protocol_config
    from robonana.configs.robotwin_flux2 import config as base
    monkeypatch.setenv("ROBONANA_PROTOCOL_SMOKE_SAVE", "1")
    monkeypatch.delenv("ROBONANA_PROTOCOL_SMOKE_STEPS", raising=False)
    with pytest.raises(ValueError, match="bounded smoke budget"):
        build_protocol_config(base, "pretrain")
    monkeypatch.setenv("ROBONANA_PROTOCOL_SMOKE_STEPS", "3")
    config = build_protocol_config(base, "pretrain")
    assert config["train"]["max_steps"] == 3
    assert not config["train"]["disable_checkpointing"]
    assert config["train"]["checkpoint_interval"] == 1
    assert config["train"]["checkpoint_save_optimizer"]
    assert config["train"]["checkpoint_keeps"] == []


def test_two_world_arms_keep_the_same_training_budget_and_data(monkeypatch):
    from robonana.configs.multitask_mbrl import build_protocol_config
    from robonana.configs.robotwin_flux2 import config as base
    monkeypatch.setenv("ROBONANA_WORLD_CONDITIONING", "fixed48")
    a = build_protocol_config(base, "pretrain")
    monkeypatch.setenv("ROBONANA_WORLD_CONDITIONING", "rope_prefix")
    b = build_protocol_config(base, "pretrain")
    assert a["optimizers"] == b["optimizers"] and a["schedulers"] == b["schedulers"]
    assert a["models"]["checkpoint"] == b["models"]["checkpoint"]
    assert a["train"]["max_steps"] == b["train"]["max_steps"] == 120000
    for cfg, mode in ((a, "fixed48"), (b, "rope_prefix")):
        assert cfg["models"]["world_conditioning"] == mode
        loader = cfg["dataloaders"]["train"]
        assert loader["data_or_config"]["world_conditioning"] == mode
        assert loader["data_or_config"]["action_chunk"] == 48
        assert loader["batch_size_per_gpu"] == 16
    assert a["dataloaders"]["train"]["data_or_config"]["task_globs"] == b["dataloaders"]["train"]["data_or_config"]["task_globs"]


@pytest.mark.parametrize("mode", ["fixed48", "rope_prefix"])
def test_world_ablation_cli_only_prints_plan(tmp_path, mode):
    import json
    import subprocess
    import sys
    from pathlib import Path
    entry = Path(__file__).resolve().parents[1] / "scripts/run_multitask_mbrl.py"
    output = tmp_path / mode
    result = subprocess.run([sys.executable, str(entry), "train", "--phase", "pretrain",
        "--world-conditioning", mode, "--output", str(output)], capture_output=True, text=True, check=True)
    config = json.loads(result.stdout)
    assert config["models"]["world_conditioning"] == mode
    assert config["dataloaders"]["train"]["data_or_config"]["world_conditioning"] == mode
    assert config["train"]["seed"] == 6666
    assert not output.exists()


def test_missing_source_rejected(monkeypatch):
    from robonana.configs.multitask_mbrl import build_protocol_config
    from robonana.configs.robotwin_flux2 import config as base
    monkeypatch.delenv("ROBONANA_MAC_PRETRAIN_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="explicit source"):
        build_protocol_config(base,"stage1")


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


def test_seed_timeout_replaced_but_locked_eval_not_replaced(tmp_path, monkeypatch):
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
    monkeypatch.setitem(sys.modules,"benchmark_robotwin_collection_pool",
                        SimpleNamespace(server_command=lambda *args:["fake-server"]))
    monkeypatch.setitem(sys.modules,"eval_robotwin_task_isolated",
                        SimpleNamespace(terminate_process_group=lambda *args,**kw:None))
    monkeypatch.setattr(module.subprocess,"Popen",lambda *args,**kw:SimpleNamespace(poll=lambda:None))
    monkeypatch.setattr(module.subprocess,"check_output",lambda *args,**kw:"revision\n")
    simulator=tmp_path/"sim"
    (simulator/"task_config").mkdir(parents=True)
    (simulator/"task_config/demo_clean.yml").write_text("config")
    opts=SimpleNamespace(command="collect",output=tmp_path/"collect",robotwin=simulator,
        checkpoint=tmp_path/"model.bin",model_config=tmp_path/"model.json",gpus=[0,1],port=8400,
        episodes=1,seed_timeout=60,seed_start=300000,candidate_multiplier=20,sim_python=Path(sys.executable),
        initial_dataset=tmp_path/"initial",manifests=None)
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
                episodes=[dict(seed=300001,success=True,hdf5=None)]))
        return 0
    monkeypatch.setattr(module,"run_bounded",fake_run)
    module.collect_lane(opts,[("hanging_mug","demo_clean")],0)
    locked_path=opts.output/"hanging_mug/demo_clean/seeds.json"
    locked=locked_path.read_bytes()
    assert prepared==[300000,300001]
    assert json.loads(locked)["jobs"][0]["seed"]==300001
    opts.command="eval"
    opts.manifests=opts.output
    opts.output=tmp_path/"eval"
    eval_commands=[]
    def error_run(command,**kwargs):
        eval_commands.append(command)
        assert "--prepare-seeds" not in command
        assert command[command.index("--capture-mode")+1]=="scout"
        return 124
    monkeypatch.setattr(module,"run_bounded",error_run)
    module.collect_lane(opts,[("hanging_mug","demo_clean")],0)
    assert len(eval_commands)==1 and locked_path.read_bytes()==locked
    summary=json.loads((opts.output/"hanging_mug/demo_clean/summary.json").read_text())
    assert summary==dict(evaluated=0,errors=1,successes=0,paired_scene_count=1)


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
    monkeypatch.setitem(sys.modules, "benchmark_robotwin_collection_pool", SimpleNamespace(server_command=lambda *a: ["server"]))
    monkeypatch.setitem(sys.modules, "eval_robotwin_task_isolated", SimpleNamespace(terminate_process_group=lambda *a, **k: None))
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
        shared_gpus=True, shard_count=8, shard_offset=0)
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
