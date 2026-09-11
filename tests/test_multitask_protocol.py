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
