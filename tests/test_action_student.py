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
