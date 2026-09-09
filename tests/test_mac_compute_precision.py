"""BF16 execution across inference, critic training, cache reuse and EMA restore."""

from unittest.mock import patch

import pytest
import torch

from test_mac_prefix_cache import model_and_inputs, sampling_inputs
from robonana.sampling import generate_mac_imaginary_rollout_h1, evaluate_mac_critics
from robonana.training.posttraining import ValueExpertEMA


def test_flow_noise_preserves_sigma_and_unrounded_velocity_target():
    from robonana.training.robotwin_trainer import flow_noise
    clean = torch.tensor([[[1., 1.003]]], dtype=torch.float32)
    sigma = torch.tensor([.999], dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16), patch(
            "torch.randn_like", side_effect=torch.zeros_like) as noise:
        noisy, target = flow_noise(clean, sigma)
    assert noise.call_args.args[0].dtype == torch.float32
    assert noisy.dtype == target.dtype == torch.float32
    assert torch.equal(noisy, clean * (1 - sigma.reshape(1, 1, 1)))
    assert noisy.bfloat16()[0, 0, 0].item() == .00099945068359375
    assert torch.equal(target, -clean)
    assert not torch.equal(target, target.bfloat16().float())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_stage1_full_world_policy_backward(device):
    from types import SimpleNamespace
    import robonana.training.robotwin_trainer as training
    RoboNanaTrainer = training.RoboNanaTrainer
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA integration requires a GPU")
    model, inputs = model_and_inputs()
    model.to(device=device, dtype=torch.bfloat16).set_training_phase("world_policy")
    trainer = object.__new__(RoboNanaTrainer)
    trainer._models = [model]
    trainer.accelerator = SimpleNamespace(device=torch.device(device))
    trainer.mixed_precision = "bf16"
    trainer.grid_height, trainer.grid_width, trainer.flow_shift = 1, 2, 1.
    trainer._posttrain_metrics = {}
    batch = dict(context=inputs["context"], context_mask=inputs["context_mask"],
                 current_latents=inputs["current_latents"], future_latents=torch.randn(2, 2, 8),
                 state=inputs["state"][:, 0], future_state=torch.randn(2, 6),
                 action=torch.randn(2, 48, 6), chunk_horizon=torch.full((2,), 48),
                 action_loss_mask=torch.tensor([1., 0.]), action_valid_mask=torch.ones(2, 48),
                 reward_chunk=torch.zeros(2, 48), reward_chunk_mask=torch.ones(2, 48),
                 success=torch.tensor([1., 0.]))
    sigma = torch.tensor([.999, .37], device=device)
    trainer._sample_timestep = lambda batch_size: sigma
    constructed = []
    def record_noise(clean, timestep):
        result = training_flow_noise(clean, timestep)
        constructed.append((clean, *result))
        return result
    training_flow_noise = training.flow_noise
    with torch.autocast(device, dtype=torch.bfloat16), \
            patch.object(training, "flow_noise", side_effect=record_noise), \
            patch.object(model, "forward", wraps=model.forward) as forward, \
            patch.object(training, "masked_mse", wraps=training.masked_mse) as world_loss, \
            patch.object(training, "masked_action_mse", wraps=training.masked_action_mse) as action_loss:
        losses = trainer._forward_step_mac_world_policy(batch)
    model_inputs = forward.call_args.kwargs
    for (clean, noisy, target), source, field in zip(
            constructed, (batch["action"], batch["future_latents"], batch["future_state"][:, None]),
            ("noisy_pred_action", "noisy_future_latents", "noisy_future_state"), strict=True):
        assert torch.equal(clean.cpu(), source)
        assert noisy.dtype == target.dtype == torch.float32
        assert torch.equal(model_inputs[field], noisy.bfloat16())
    assert torch.equal(model_inputs["gt_action_cond"], batch["action"].to(device).bfloat16())
    assert torch.equal(model_inputs["action_timestep"], sigma)
    assert torch.equal(model_inputs["wm_timestep"], sigma)
    # Verify actual loss inputs, not only the helper's intermediate tensors.
    assert torch.equal(action_loss.call_args.args[1], constructed[0][2])
    assert torch.equal(world_loss.call_args_list[0].args[1], constructed[1][2])
    assert torch.equal(world_loss.call_args_list[1].args[1], constructed[2][2])
    assert all(loss.dtype == torch.float32 and torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()
    for name in ("action_out.weight", "state_out.weight", "reward_out.weight", "success_out.weight"):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert all(p.grad is None for p in model.value_expert.parameters())
    assert all(p.grad is None for p in model.q_expert.parameters())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_imagination_regression_and_target_restore(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA integration requires a GPU")
    model, inputs = model_and_inputs()
    model.to(device=device, dtype=torch.bfloat16).set_training_phase("critic")
    inputs = {key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
              for key, value in inputs.items()}
    target = ValueExpertEMA(model.value_expert)
    with patch.object(model, "prefill_condition_cache", wraps=model.prefill_condition_cache) as prefill:
        rollout = generate_mac_imaginary_rollout_h1(
            online_model=model, target_value_expert=target.model,
            **sampling_inputs(inputs), candidate_count=2,
            action_noise=torch.randn(2, 2, 48, 6, device=device, dtype=torch.bfloat16),
            future_noise=torch.randn(2, 2, 8, device=device, dtype=torch.bfloat16),
            future_state_noise=torch.randn(2, 1, 6, device=device, dtype=torch.bfloat16),
            schedule=torch.tensor([1., .5, 0.], device=device), discount=.999,
            reward_non_goal=-1., reward_goal=0., return_scale=1000.,
            grid_height=1, grid_width=2)
        assert prefill.call_count == 2
    assert rollout.condition_cache.compute_dtype == torch.bfloat16
    assert rollout.reward_logits.dtype == torch.bfloat16
    assert rollout.value_target_return.dtype == torch.float32  # Stable target arithmetic.
    # FACT/Accelerate autocast training must reuse direct-BF16 inference caches.
    with torch.autocast(device, dtype=torch.bfloat16), patch.object(
            model, "prefill_condition_cache", side_effect=AssertionError("C recomputed")):
        value, q = evaluate_mac_critics(
            model=model, **sampling_inputs(inputs), clean_action=rollout.selected_action,
            condition_cache=rollout.condition_cache, grid_height=1, grid_width=2)
    assert value.dtype == q.dtype == torch.bfloat16
    (value.float().square().mean() + q.float().square().mean()).backward()
    for parameter in model.parameters():
        assert parameter.dtype == torch.bfloat16
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None
    target.update(model.value_expert, optimizer_step=1, optimizer_step_succeeded=True)
    saved = target.state_dict()
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in target.model.parameters())
    target.load_state_dict(saved)
    for name, parameter in target.model.state_dict().items():
        assert torch.equal(parameter.cpu(), saved[name])


def test_bf16_ema_single_interpolation_and_fp32_checkpoint_load():
    online = torch.nn.Linear(2, 1, bias=False).bfloat16()
    online.weight.data.fill_(1)
    target = ValueExpertEMA(online, decay=.995)
    for step in range(3):
        target.update(online, optimizer_step=step, optimizer_step_succeeded=True)
    assert torch.equal(target.model.weight, online.weight)
    online.weight.data.fill_(2)
    expected = torch.lerp(target.model.weight, online.weight, .005)
    target.update(online, optimizer_step=4, optimizer_step_succeeded=True)
    assert torch.equal(target.model.weight, expected)
    assert not torch.equal(target.model.weight, torch.ones_like(expected))
    target.load_state_dict({"weight": torch.full((1, 2), 3., dtype=torch.float32)})
    assert target.model.weight.dtype == torch.bfloat16
    assert torch.equal(target.model.weight, torch.full_like(expected, 3))
