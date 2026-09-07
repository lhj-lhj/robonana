"""The MAC model has one execution precision; external encoders are separate."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from test_mac_prefix_cache import model_and_inputs, sampling_inputs
from robonana.models.pretrained import load_flux2_fact_trained_checkpoint
from robonana.sampling import generate_mac_imaginary_rollout_h1, evaluate_mac_critics
from robonana.training.posttraining import ValueExpertEMA, fp32_compute_context


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp32_imagination_and_regression_reuse_current_cache(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA precision check requires a GPU")
    model, inputs = model_and_inputs()
    model.to(device=device).set_training_phase("critic")
    inputs = {key: value.to(device=device) for key, value in inputs.items()}
    target = ValueExpertEMA(model.value_expert)
    with fp32_compute_context(model):
        assert not torch.is_autocast_enabled(device)
        assert model.cache_compute_dtype() == torch.float32
        with patch.object(model, "prefill_condition_cache", wraps=model.prefill_condition_cache) as prefill:
            rollout = generate_mac_imaginary_rollout_h1(
                online_model=model, target_value_expert=target.model,
                **sampling_inputs(inputs), candidate_count=2,
                action_noise=torch.randn(2, 2, 48, 6, device=device),
                future_noise=torch.randn(2, 2, 8, device=device),
                future_state_noise=torch.randn(2, 1, 6, device=device),
                schedule=torch.tensor([1., .5, 0.], device=device), discount=.999,
                reward_non_goal=-1., reward_goal=0., return_scale=1000.,
                grid_height=1, grid_width=2)
            assert prefill.call_count == 2  # Current C and next C only.
        assert rollout.condition_cache.compute_dtype == torch.float32
        assert rollout.reward_logits.dtype == torch.float32
        assert rollout.value_target_return.dtype == torch.float32
    with fp32_compute_context(model), patch.object(
            model, "prefill_condition_cache", side_effect=AssertionError("C recomputed")):
        value, q = evaluate_mac_critics(
            model=model, **sampling_inputs(inputs), clean_action=rollout.selected_action,
            condition_cache=rollout.condition_cache, grid_height=1, grid_width=2)
    (value.square().mean() + q.square().mean()).backward()
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in target.model.parameters())
    for parameter in model.parameters():
        assert parameter.dtype == torch.float32
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None


@pytest.mark.parametrize("precision", ["bf16", "fp16", "fp8"])
def test_config_rejects_retired_precision_overrides(monkeypatch, precision):
    import importlib
    import sys
    monkeypatch.setenv("ROBONANA_MIXED_PRECISION", precision)
    name = "robonana.configs.robotwin_flux2"
    sys.modules.pop(name, None)
    try:
        with pytest.raises(ValueError, match="FP32-only"):
            importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)


def test_non_fp32_weights_and_ambient_autocast_are_rejected():
    model, inputs = model_and_inputs()
    # Reject unsupported precision before checkpoint IO or an expensive pass.
    with pytest.raises(ValueError, match="FP32"):
        load_flux2_fact_trained_checkpoint("not-opened.bin", dtype=torch.float64)
    model.double()
    with pytest.raises(ValueError, match="FP32"):
        fp32_compute_context(model)
    with pytest.raises(ValueError, match="FP32"):
        model.prefill_condition_cache(**inputs)
    model.float()
    with torch.autocast("cpu"):
        with pytest.raises(ValueError, match="autocast disabled"):
            model.prefill_condition_cache(**inputs)
        with fp32_compute_context(model):
            assert model.prefill_condition_cache(**inputs).compute_dtype == torch.float32


def test_inference_entrypoints_have_no_dtype_switch():
    root = Path(__file__).resolve().parents[1]
    for name in ("inference_server_robotwin.py", "inference_server_robotwin_batched.py",
                 "inference_server_robotwin_xpolicylab.py", "eval_robotwin_all_tasks_parallel.sh"):
        assert "--dtype" not in (root / "scripts" / name).read_text(encoding="utf-8")
