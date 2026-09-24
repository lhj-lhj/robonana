"""V3 joint MoT: attached gradients, label isolation and one inference path."""
from dataclasses import asdict, replace
import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from test_mac_prefix_cache import model_and_inputs, sampling_inputs, full_action
from test_world_rope_prefix import world_inputs
from robonana.models.mac_flux2_fact import MacFlux2FACTModel
from robonana.models.pretrained import load_flux2_backbone_checkpoint, load_flux2_fact_trained_checkpoint
from robonana.training.optimizer import build_optimizer_param_groups
from robonana.sampling import sample_flux2_action, sample_mac_world, sample_action_flow


def setup_v3(device="cpu"):
    base, inputs = model_and_inputs()
    # Keep a multi-layer backbone: a single layer cannot catch indirect leakage.
    from test_pretrained import tiny_params
    params = replace(tiny_params(), depth=2, depth_single_blocks=2)
    model = MacFlux2FACTModel(params, action_dim=6, state_dim=6,
                             expert_hidden_dim=16, architecture_version="mac_mot_v3")
    model.set_training_phase("world_policy")
    kwargs = world_inputs(inputs, torch.full((2,), 48))
    return model.to(device), {k: v.to(device) for k, v in kwargs.items()}, params


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_action_loss_trains_shared_backbone_without_label_leak(checkpointing, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model, inputs, _ = setup_v3(device)
    if checkpointing:
        model.enable_gradient_checkpointing()
    for name in ("context", "current_latents", "state", "gt_action_cond", "noisy_future_latents", "noisy_future_state"):
        inputs[name].requires_grad_()
    out = model(**inputs)
    assert out.action.shape == (2, 48, 6)
    assert out.segments.pred_action.start == out.segments.pred_action.stop
    out.action.square().mean().backward()
    for name in ("context", "current_latents", "state"):
        assert inputs[name].grad.abs().sum() > 0
    for name in ("gt_action_cond", "noisy_future_latents", "noisy_future_state"):
        grad = inputs[name].grad
        assert grad is None or torch.count_nonzero(grad) == 0
    for parameter in (model.txt_in.weight, model.img_in.weight,
                      model.double_blocks[0].img_attn.qkv.weight,
                      model.single_blocks[0].linear1.weight,
                      model.action_expert.action_encoder.weight,
                      model.action_expert.head.linear.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.value_expert.parameters())


def test_checkpointing_preserves_joint_outputs_and_gradients():
    first, inputs, _ = setup_v3()
    second = copy.deepcopy(first)
    second.enable_gradient_checkpointing()
    second.set_gradient_checkpointing_single_stride(2)
    for model in (first, second):
        out = model(**inputs)
        sum(getattr(out, k).square().mean() for k in
            ("action", "image", "future_state", "reward", "success")).backward()
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)


def test_world_and_action_are_independent_of_each_others_targets():
    model, inputs, _ = setup_v3()
    model.eval()
    with torch.no_grad():
        baseline = model(**inputs)
        changed = model(**{**inputs, "noisy_pred_action": inputs["noisy_pred_action"] + 100})
        for key in ("image", "future_state", "reward", "success"):
            torch.testing.assert_close(getattr(baseline, key), getattr(changed, key), atol=0, rtol=0)
        altered = {**inputs}
        for key in ("gt_action_cond", "noisy_future_latents", "noisy_future_state"):
            altered[key] = inputs[key] + 100
        torch.testing.assert_close(model(**altered).action, baseline.action, atol=0, rtol=0)
        assert not torch.equal(model(**{**inputs, "action_timestep": torch.zeros(2)}).action, baseline.action)


def test_joint_action_matches_cached_candidate_mapping_and_full_sampler():
    model, inputs, _ = setup_v3()
    model.eval()
    condition = {k: inputs[k] for k in ("context", "context_ids", "current_latents", "current_ids", "state", "context_mask")}
    cache = model.prefill_condition_cache(**condition)
    with torch.no_grad():
        full = model(**inputs).action
        cached = model.predict_action_cached(cache, inputs["noisy_pred_action"],
                    batch_indices=torch.arange(2), timestep=inputs["action_timestep"])
        torch.testing.assert_close(full, cached, atol=3e-6, rtol=3e-5)
        indices = torch.tensor([1, 0, 1])
        reordered = model.predict_action_cached(cache, inputs["noisy_pred_action"][indices],
                    batch_indices=indices, timestep=inputs["action_timestep"][indices])
        torch.testing.assert_close(reordered, full[indices], atol=3e-6, rtol=3e-5)
        args = dict(model=model, **sampling_inputs(condition),
                    action_noise=inputs["noisy_pred_action"], schedule=torch.linspace(1, 0, 21),
                    chunk_horizon=48, grid_height=1, grid_width=2)
        a = sample_flux2_action(**args)
        b = sample_action_flow(action_noise=inputs["noisy_pred_action"], schedule=args["schedule"],
                               predict_action=lambda action, sigma: full_action(model, condition, action, sigma))
        torch.testing.assert_close(a, b, atol=5e-6, rtol=5e-5)
        world_args = dict(model=model, **sampling_inputs(condition), clean_action=inputs["gt_action_cond"],
                          future_noise=inputs["noisy_future_latents"], future_state_noise=inputs["noisy_future_state"],
                          schedule=torch.linspace(1, 0, 3), grid_height=1, grid_width=2)
        world_cached = sample_mac_world(**world_args)
        world_full = sample_mac_world(**world_args, use_cache=False)
        for key in ("future", "future_state", "reward_logits", "success_logit"):
            torch.testing.assert_close(getattr(world_cached, key), getattr(world_full, key), atol=3e-6, rtol=3e-5)


def test_action_expert_uses_robot_lr_and_freezes_in_critic_phase():
    model, _, _ = setup_v3()
    groups = build_optimizer_param_groups(model, base_lr=2e-5, robot_lr=1e-4)
    robot = next(g for g in groups if g["name"] == "robot_modules")
    assert robot["lr"] == 1e-4
    assert {id(p) for p in model.action_expert.parameters()} <= {id(p) for p in robot["params"]}
    all_ids = [id(p) for g in groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids)) == sum(p.requires_grad for p in model.parameters())
    model.set_training_phase("critic")
    assert all(not p.requires_grad for p in model.action_expert.parameters())
    assert not model.action_expert.training
    assert all(p.requires_grad for p in model.value_expert.parameters())


def test_original_flux_initialization_and_strict_v3_checkpoint_roundtrip(tmp_path):
    from flux2.model import Flux2
    _, inputs, params = setup_v3()
    flux = Flux2(params)
    path = tmp_path / "flux.safetensors"
    save_file(flux.state_dict(), path)
    model, _ = load_flux2_backbone_checkpoint(path, params=params, action_dim=6, state_dim=6,
                expert_hidden_dim=16, architecture_version="mac_mot_v3", dtype=torch.float32)
    for name, value in flux.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    assert "action_out.weight" not in model.state_dict()
    checkpoint = tmp_path / "model.bin"
    torch.save(model.state_dict(), checkpoint)
    config = dict(params=asdict(params), action_dim=6, state_dim=6, reward_dim=48,
                  success_dim=1, q_dim=1, value_dim=1, max_horizon=48, chunk_horizon=48,
                  reward_head_type="binary_chunk", architecture_version="mac_mot_v3",
                  expert_hidden_dim=16, world_conditioning="fixed48")
    (tmp_path / "model_config.json").write_text(json.dumps(config))
    restored, _ = load_flux2_fact_trained_checkpoint(checkpoint, device="cpu", dtype=torch.float32,
                                                    architecture_version="mac_mot_v3")
    torch.testing.assert_close(restored(**inputs).action, model(**inputs).action, atol=0, rtol=0)
    with pytest.raises(ValueError, match="architecture_version"):
        load_flux2_fact_trained_checkpoint(checkpoint, device="cpu", architecture_version="mac_mot_v2")
    # Checkpoint labels cannot silently convert a v2 state dict into v3.
    v2 = MacFlux2FACTModel(params, action_dim=6, state_dim=6, expert_hidden_dim=16)
    torch.save(v2.state_dict(), checkpoint)
    with pytest.raises(RuntimeError):
        load_flux2_fact_trained_checkpoint(checkpoint, device="cpu")


def test_v3_config_is_explicit_and_preserves_data_and_loss_contract():
    from robonana.configs.schema import load_options
    from robonana.configs.training import TrainOptions, build_training_config
    root = Path(__file__).resolve().parents[1]
    v3 = load_options(TrainOptions, root / "configs/train_v3.json")
    assert v3.global_batch == len(v3.gpus) * v3.microbatch * v3.accumulation_steps == 256
    assert v3.sampling_steps == 20 and v3.max_steps == 120000
    a = build_training_config(v3)
    b = build_training_config(replace(v3, architecture_version="mac_mot_v2"))
    assert a["dataloaders"] == b["dataloaders"]
    assert a["optimizers"] == b["optimizers"] and a["schedulers"] == b["schedulers"]
    assert a["train"] == b["train"]
    with pytest.raises(ValueError, match="fixed48"):
        replace(v3, world_conditioning="rope_prefix")


def test_cuda_bfloat16_joint_and_cached_action():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model, inputs, _ = setup_v3("cuda")
    model.bfloat16().enable_gradient_checkpointing()
    inputs = {k: v.bfloat16() if v.is_floating_point() and "timestep" not in k else v
              for k, v in inputs.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inputs)
        loss = sum(getattr(out, name).float().square().mean()
                   for name in ("action", "image", "future_state", "reward", "success"))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.action_expert.head.linear.weight.grad.abs().sum() > 0
    condition = {k: inputs[k] for k in ("context", "context_ids", "current_latents", "current_ids", "state", "context_mask")}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        cache = model.prefill_condition_cache(**condition)
        actual = model.predict_action_cached(cache, inputs["noisy_pred_action"],
                    batch_indices=torch.arange(2, device="cuda"), timestep=inputs["action_timestep"])
    torch.testing.assert_close(actual, out.action, atol=.025, rtol=.025)


def test_v3_adam_resume_matches_uninterrupted_update():
    model, inputs, _ = setup_v3()
    model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(build_optimizer_param_groups(model, base_lr=2e-5, robot_lr=1e-4))
    def step(m, opt):
        opt.zero_grad(set_to_none=True)
        out = m(**inputs)
        sum(getattr(out, name).square().mean() for name in
            ("action", "image", "future_state", "reward", "success")).backward()
        opt.step()
    step(model, optimizer)
    restored = copy.deepcopy(model)
    restored_opt = torch.optim.AdamW(build_optimizer_param_groups(restored, base_lr=2e-5, robot_lr=1e-4))
    restored_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model, optimizer)
    step(restored, restored_opt)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], atol=0, rtol=0)


def _ddp_v3_worker(rank, rendezvous):
    import torch.distributed as dist
    from datetime import timedelta
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous,
                            timeout=timedelta(seconds=90))
    try:
        model, inputs, _ = setup_v3()
        model.enable_gradient_checkpointing()
        ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        optimizer = torch.optim.AdamW(build_optimizer_param_groups(model, base_lr=2e-5, robot_lr=1e-4))
        # Different batches exercise synchronization; two iterations catch an
        # unfinished reducer from any mistakenly registered unused v3 parameter.
        inputs["context"] = inputs["context"] + rank
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            out = ddp(**inputs)
            sum(getattr(out, name).square().mean() for name in
                ("action", "image", "future_state", "reward", "success")).backward()
            optimizer.step()
        for parameter in (model.img_in.weight, model.action_expert.head.linear.weight):
            copies = [torch.empty_like(parameter) for _ in range(2)]
            dist.all_gather(copies, parameter.detach())
            torch.testing.assert_close(copies[0], copies[1], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_v3_two_rank_training_has_no_unused_reducer_parameters(tmp_path):
    torch.multiprocessing.spawn(_ddp_v3_worker,
        args=((tmp_path / "v3-rendezvous").as_uri(),), nprocs=2, join=True)
