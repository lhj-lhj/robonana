"""Isolated action expert: same-noise target, no backbone or critic mutation."""
import torch
from test_mac_prefix_cache import model_and_inputs
from robonana.models.flux2_action_student import build_action_student


def test_student_reuses_blocks_and_updates_only_itself():
    teacher, inputs = model_and_inputs()
    teacher.requires_grad_(False)
    original = {k:v.clone() for k,v in teacher.state_dict().items()}
    student = build_action_student(teacher, hidden_dim=16)
    cache = teacher.prefill_condition_cache(**inputs)
    noise = torch.randn(2,48,6)
    ids = teacher._robot_ids(batch_size=2,length=48,segment_id=3,device='cpu',dtype=torch.long,
                            time_ids=torch.arange(1,49)[None].expand(2,-1))
    pe = teacher.pe_embedder(ids)
    out = student(cache,noise=noise,query_pe=pe)
    assert out.shape == (2,48,6)
    assert not torch.equal(out,student(cache,noise=noise+1,query_pe=pe))
    assert not any(k.startswith('query.') for k in student.state_dict())
    out.square().mean().backward()
    assert student.action_encoder.weight.grad is not None
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(v,original[k]) for k,v in teacher.state_dict().items())
    assert torch.isfinite(out).all()


def test_universal_adam_audit_rejects_reset_step():
    import pytest
    from robonana.training.continuation import validate_universal_adam
    parameter=torch.nn.Parameter(torch.ones(2))
    opt=torch.optim.AdamW([parameter])
    parameter.sum().backward();opt.step()
    state=opt.state[parameter]['exp_avg'].clone()
    assert validate_universal_adam(opt,1)==1
    assert torch.equal(opt.state[parameter]['exp_avg'],state)
    with pytest.raises(ValueError,match='step mismatch'):
        validate_universal_adam(opt,4000)


def test_stage1_only_excludes_both_pretrain_splits():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1]/'scripts/diagnostics/compare_stage1_policy.py'
    spec = importlib.util.spec_from_file_location('stage1_comparison',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(stage1_only=True,stage1='teacher',stage1_config='config',
                           pretrain='baseline',pretrain_config='baseline_config')
    assert module.evaluation_policies(args) == [('stage1','teacher','config')]
    args.stage1_only = False
    assert len(module.evaluation_policies(args)) == 2


def test_episode_split_is_stable_for_all_windows():
    # Execute only the pure split helper: no training or heavy imports.
    import ast
    import hashlib
    from pathlib import Path
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[1]/'scripts/diagnostics/train_action_student.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
              if isinstance(n,ast.FunctionDef) and n.name=='heldout_episode')
    scope={'hashlib':hashlib}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    for episode in range(50):
        records=[SimpleNamespace(source=Path('same_source'),episode_index=episode,window=t)
                 for t in range(10)]
        assert len({scope['heldout_episode'](r) for r in records})==1


def test_student_extended_schedule_preserves_adam_state():
    import ast
    import math
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/diagnostics/train_action_student.py'
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body
           if isinstance(n,ast.FunctionDef) and n.name in
           ('student_lr_multiplier','restore_student_schedule')]
    scope={'math':math}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    parameter=torch.nn.Parameter(torch.ones(2))
    opt=torch.optim.AdamW([parameter],lr=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,
        lambda step:scope['student_lr_multiplier'](step,20000))
    parameter.sum().backward();opt.step()
    moments=opt.state[parameter]['exp_avg'].clone()
    adam_step=opt.state[parameter]['step'].clone()
    opt.param_groups[0]['lr']=6.8e-7
    scope['restore_student_schedule'](scheduler,opt,1500,2000,20000)
    assert 9.8e-5 < opt.param_groups[0]['lr'] < 1e-4
    assert torch.equal(moments,opt.state[parameter]['exp_avg'])
    assert torch.equal(adam_step,opt.state[parameter]['step'])
    assert scope['student_lr_multiplier'](20000,20000)==0
    opt.param_groups[0]['lr']=3e-5
    scope['restore_student_schedule'](scheduler,opt,1500,20000,20000)
    assert opt.param_groups[0]['lr']==3e-5
