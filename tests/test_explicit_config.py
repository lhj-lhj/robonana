"""显式配置合同：缺信息立即报错；多处使用的量只有一个来源。"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import pytest
from robonana.configs.schema import load_options
from robonana.configs.training import TrainOptions, build_training_config
from robonana.configs.evaluation import EvalOptions
from robonana.configs.resume import ResumeOptions, build_resume_config
from robonana.normalization import A_STATS_PATH

ROOT = Path(__file__).resolve().parents[1]


def options(tmp_path, **kw):
    base=load_options(TrainOptions,ROOT/'configs/train.json')
    args=dict(output=tmp_path/'run',dataset_root=tmp_path/'data',flux_checkpoint_dir=tmp_path/'flux',
              stats_path=A_STATS_PATH,checkpoint_keeps=())
    args.update(kw)
    return replace(base,**args)



@pytest.mark.parametrize('phase,steps,lr,robot_lr', [('pretrain',120000,2e-5,1e-4),('stage1',60000,2e-5,2e-5),('stage2',20000,1e-4,1e-4)])
def test_phases_share_one_builder_and_actual_dataset_contract(tmp_path, phase, steps, lr, robot_lr):
    extra = {} if phase=='pretrain' else dict(checkpoint=tmp_path/'source.bin', model_config=tmp_path/'source.json', replay_root=tmp_path/'replay')
    o=options(tmp_path,phase=phase,max_steps=steps,lr=lr,robot_lr=robot_lr,**extra)
    config=build_training_config(o)
    assert config['train']['max_steps']==config['schedulers']['decay_steps']==steps
    assert config['train']['checkpoint_keeps'][-1]==steps
    assert config['optimizers']['lr']==lr and config['optimizers']['robot_lr']==robot_lr
    assert config['train']['loss_weights']['reward_loss']==.1
    assert config['models']['train_mode']==('critic' if phase=='stage2' else 'world_policy')
    from robonana.data.robotwin_hdf5 import RoboTwinHDF5Dataset
    from robonana.data.robotwin_lerobot import RoboTwinLeRobotDataset
    specs=config['dataloaders']['train']['data_or_config']
    specs=specs if isinstance(specs,list) else [specs]
    classes={c.__name__:c for c in (RoboTwinHDF5Dataset,RoboTwinLeRobotDataset)}
    for spec in specs:
        assert ('task_globs' in spec) == (spec['_class_name']=='RoboTwinLeRobotDataset')
        classes[spec['_class_name']].load(spec).close()


@pytest.mark.parametrize('gpus,micro,acc,total', [((0,1,2,3),16,2,128),((0,1,2,3),32,2,256),(tuple(range(8)),32,1,256)])
def test_batch_is_explicit_and_consistent(tmp_path,gpus,micro,acc,total):
    o=options(tmp_path,gpus=gpus,microbatch=micro,accumulation_steps=acc,global_batch=total)
    c=build_training_config(o)
    assert len(c['launch']['gpu_ids'])*c['dataloaders']['train']['batch_size_per_gpu']*c['train']['gradient_accumulation_steps']==total
    with pytest.raises(ValueError,match='equal global_batch'):
        replace(o,accumulation_steps=acc+1)


def test_no_import_time_environment_or_shared_mutation(tmp_path, monkeypatch):
    o=options(tmp_path)
    a=build_training_config(o)
    monkeypatch.setenv('ROBONANA_MAX_STEPS','3')
    monkeypatch.setenv('ROBONANA_WORLD_CONDITIONING','rope_prefix')
    assert build_training_config(o)==a
    b=build_training_config(replace(o,world_conditioning='rope_prefix',max_steps=100000))
    assert b['models']['world_conditioning']==b['dataloaders']['train']['data_or_config']['world_conditioning']=='rope_prefix'
    assert b['train']['max_steps']==b['schedulers']['decay_steps']==b['train']['checkpoint_keeps'][-1]==100000
    a['models']['params']['axes_dim'][0]=999
    assert build_training_config(o)['models']['params']['axes_dim'][0]==32


@pytest.mark.parametrize('filename,cls',[('train.json',TrainOptions),('eval.json',EvalOptions),('resume.json',ResumeOptions)])
def test_examples_are_complete_and_paths_anchor_to_config(filename,cls):
    o=load_options(cls,ROOT/'configs'/filename)
    assert o.output.is_absolute()
    assert o.output.parent==ROOT/('outputs' if filename=='eval.json' else 'experiments')


@pytest.mark.parametrize('key', ['gpus','microbatch','accumulation_steps','global_batch','max_steps','warmup_steps','lr','robot_lr','world_conditioning'])
def test_missing_information_never_falls_back(tmp_path,key):
    raw=json.loads((ROOT/'configs/train.json').read_text());raw.pop(key)
    path=tmp_path/'missing.json';path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='Missing explicit config fields'):load_options(TrainOptions,path)


@pytest.mark.parametrize('key,value',[('microbatch',True),('global_batch','128'),('gradient_checkpointing','false'),('gpus',[0,0]),('warmup_steps',120001),('typo_steps',2)])
def test_bad_types_typos_and_conflicts_fail(tmp_path,key,value):
    raw=json.loads((ROOT/'configs/train.json').read_text());raw[key]=value
    path=tmp_path/'bad.json';path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):load_options(TrainOptions,path)


def test_cli_displays_exact_plan_without_writing(tmp_path):
    raw=json.loads((ROOT/'configs/train.json').read_text());raw['output']=str(tmp_path/'not-created')
    path=tmp_path/'input.json';path.write_text(json.dumps(raw))
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_multitask_mbrl.py'),'train','--config',str(path)],capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan['batch']==dict(gpus=8,microbatch=16,accumulation_steps=1,global_batch=128)
    assert plan['resolved']['train']['max_steps']==120000
    assert not (tmp_path/'not-created').exists()


def test_fact_json_roundtrip_registers_both_datasets(tmp_path):
    from fact_train import Config,load_config
    from robonana.training import robotwin_trainer
    c=build_training_config(options(tmp_path))
    path=tmp_path/'config.json';Config(c).save(str(path))
    loaded=load_config(str(path))
    assert loaded.optimizers.betas == (.9,.95)
    assert loaded.train.gradient_accumulation_steps==1


def test_resume_keeps_saved_algorithm_and_requires_consistent_budget(tmp_path):
    source=build_training_config(options(tmp_path))
    path=tmp_path/'saved.json';path.write_text(json.dumps(source))
    o=ResumeOptions(output=tmp_path/'new',checkpoint=tmp_path/'checkpoint',model_config=path,
        gpus=tuple(range(8)),microbatch=16,accumulation_steps=1,global_batch=128,
        gradient_checkpointing=True,single_checkpoint_stride=4,additional_steps=20000,max_steps=140000,universal_checkpoint=False)
    c=build_resume_config(o)
    assert c['train']['max_steps']==140000 and c['schedulers']['decay_steps']==20000
    assert c['train']['posttrain']==source['train']['posttrain']
    with pytest.raises(ValueError,match='contradicts'):build_resume_config(replace(o,max_steps=150000))


def test_sim_python_keeps_virtualenv_symlink(tmp_path):
    raw=json.loads((ROOT/'configs/eval.json').read_text())
    target=tmp_path/'system-python';target.touch()
    executable=tmp_path/'venv/bin/python';executable.parent.mkdir(parents=True);executable.symlink_to(target)
    raw['sim_python']='venv/bin/python'
    path=tmp_path/'eval.json';path.write_text(json.dumps(raw))
    assert load_options(EvalOptions,path).sim_python == executable


def test_explicit_smoke_updates_budget_scheduler_and_retention(tmp_path):
    o=options(tmp_path,smoke_steps=3,smoke_save=True,checkpoint_keeps=(10000,120000))
    c=build_training_config(o)
    assert c['train']['max_steps']==c['schedulers']['decay_steps']==3
    assert c['schedulers']['warmup_steps']==1
    assert c['train']['checkpoint_interval']==1 and c['train']['checkpoint_keeps']==[]
    assert not c['train']['disable_checkpointing']


def test_ignored_legacy_environment_cannot_override_explicit_batch(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBONANA_BATCH_SIZE','999')
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_multitask_mbrl.py'),'train','--config',str(ROOT/'configs/train.json')],capture_output=True,text=True)
    # 用户已关闭旧环境变量的启动拒绝；仍必须保证它不能覆盖显式 JSON。
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['batch']['global_batch'] == plan['requested']['global_batch'] == 128
