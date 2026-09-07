"""One FLUX precision authority for imagination, online critics and target V."""

from unittest.mock import patch

import pytest
import torch

from test_mac_prefix_cache import model_and_inputs, sampling_inputs
from robonana.sampling import generate_mac_imaginary_rollout_h1, evaluate_mac_critics
from robonana.training.posttraining import ValueExpertEMA, flux_compute_context


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_imagination_and_regression_follow_flux_and_reuse_current_cache(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA precision check requires a GPU")
    model, inputs = model_and_inputs()
    model.to(device=device, dtype=dtype).set_training_phase("critic")
    inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
              for key, value in inputs.items()}
    target = ValueExpertEMA(model.value_expert)
    # FP32 FLUX must override even an ambient BF16 context; BF16 FLUX must
    # autocast the FP32-stored target expert without converting its weights.
    with torch.autocast(device, dtype=torch.bfloat16):
        with flux_compute_context(model):
            assert torch.is_autocast_enabled(device) == (dtype == torch.bfloat16)
            assert model.cache_compute_dtype() == dtype
            with patch.object(model, "prefill_condition_cache", wraps=model.prefill_condition_cache) as prefill:
                rollout = generate_mac_imaginary_rollout_h1(
                    online_model=model, target_value_expert=target.model,
                    **sampling_inputs(inputs), candidate_count=2,
                    action_noise=torch.randn(2, 2, 48, 6, device=device).to(dtype),
                    future_noise=torch.randn(2, 2, 8, device=device).to(dtype),
                    future_state_noise=torch.randn(2, 1, 6, device=device).to(dtype),
                    schedule=torch.tensor([1., .5, 0.], device=device), discount=.999,
                    reward_non_goal=-1., reward_goal=0., return_scale=1000.,
                    grid_height=1, grid_width=2)
                assert prefill.call_count == 2  # Current C and next C only.
            assert rollout.condition_cache.compute_dtype == dtype
            assert rollout.reward_logits.dtype == dtype
            assert rollout.value_target_return.dtype == torch.float32
        with flux_compute_context(model), patch.object(
                model, "prefill_condition_cache", side_effect=AssertionError("C recomputed")):
            value, q = evaluate_mac_critics(
                model=model, **sampling_inputs(inputs), clean_action=rollout.selected_action,
                condition_cache=rollout.condition_cache, grid_height=1, grid_width=2)
    (value.float().square().mean() + q.float().square().mean()).backward()
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in target.model.parameters())
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None


@pytest.mark.parametrize("precision", ["no", "bf16"])
def test_config_has_one_explicit_precision_override(monkeypatch, precision):
    import importlib
    import sys
    monkeypatch.setenv("ROBONANA_MIXED_PRECISION", precision)
    name = "robonana.configs.robotwin_flux2"
    sys.modules.pop(name, None)
    try:
        assert importlib.import_module(name).config["train"]["mixed_precision"] == precision
    finally:
        sys.modules.pop(name, None)
