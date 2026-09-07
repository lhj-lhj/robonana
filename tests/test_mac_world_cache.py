"""Full FLUX (not another cached path) is the world-cache numerical oracle."""

from unittest.mock import patch

import pytest
import torch

from test_mac_prefix_cache import model_and_inputs, sampling_inputs
from robonana.models.position_ids import image_position_ids
from robonana.sampling import evaluate_mac_critics, sample_mac_world


@pytest.mark.parametrize("autocast", [False, True])
def test_twenty_step_world_cache_matches_full_flow_and_logits(autocast):
    model, inputs = model_and_inputs()
    kwargs = dict(model=model, **sampling_inputs(inputs), clean_action=torch.randn(2, 48, 6),
                  future_noise=torch.randn(2, 2, 8), future_state_noise=torch.randn(2, 1, 6),
                  schedule=torch.linspace(1, 0, 21), grid_height=1, grid_width=2)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        reference = sample_mac_world(**kwargs, use_cache=False)
        cache = model.prefill_condition_cache(**inputs)
        with patch.object(model, "prefill_condition_cache", side_effect=AssertionError("C recomputed")), \
             patch.object(model, "forward", side_effect=AssertionError("full world forward")), \
             patch.object(model, "predict_world_cached", wraps=model.predict_world_cached) as predict:
            actual = sample_mac_world(**kwargs, condition_cache=cache)
            assert predict.call_count == 20
    for field in ("future", "future_state", "reward_logits", "success_logit"):
        # BF16 changes GEMM/SDPA shapes and therefore rounding, not equations.
        torch.testing.assert_close(getattr(actual, field), getattr(reference, field),
                                   atol=0.025 if autocast else 3e-6,
                                   rtol=0.025 if autocast else 3e-5)


def test_world_cache_preserves_cascade_and_is_detached():
    model, inputs = model_and_inputs()
    cache = model.prefill_world_cache(
        condition_cache=model.prefill_condition_cache(**inputs), clean_action=torch.randn(2, 48, 6),
        language_length=3, state_length=1, image_length=2,
        future_state_length=1, future_image_length=2, context_mask=inputs["context_mask"])
    args = dict(noisy_future_latents=torch.randn(2, 2, 8), noisy_future_state=torch.randn(2, 1, 6),
                future_ids=image_position_ids(2, grid_height=1, grid_width=2,
                    time_coord=torch.full((2,), 48), device="cpu"), wm_timestep=torch.full((2,), 0.7))
    baseline = model.predict_world_cached(cache, **args)
    changed_image = model.predict_world_cached(cache, **{**args, "noisy_future_latents": args["noisy_future_latents"] + 10})
    changed_state = model.predict_world_cached(cache, **{**args, "noisy_future_state": args["noisy_future_state"] + 10})
    torch.testing.assert_close(baseline.future_state, changed_image.future_state, rtol=0, atol=0)
    assert not torch.allclose(baseline.image, changed_state.image)
    for output in (changed_image, changed_state):
        assert torch.equal(output.reward, baseline.reward)
        assert torch.equal(output.success, baseline.success)
    assert all(not tensor.requires_grad for layer in (*cache.kv.double, *cache.kv.single)
               for tensor in layer.values())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="precision"):
            model.predict_world_cached(cache, **args)


@pytest.mark.parametrize("sampling_bf16", [False, True])
def test_critic_reuses_only_same_precision_cache_with_identical_gradients(sampling_bf16):
    model, inputs = model_and_inputs()
    model.set_training_phase("critic")
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=sampling_bf16):
        cache = model.prefill_condition_cache(**inputs)
    kwargs = dict(model=model, **sampling_inputs(inputs), clean_action=torch.randn(2, 48, 6),
                  grid_height=1, grid_width=2)
    reference = evaluate_mac_critics(**kwargs)
    sum(value.square().mean() for value in reference).backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    with patch.object(model, "prefill_condition_cache", wraps=model.prefill_condition_cache) as prefill:
        actual = evaluate_mac_critics(**kwargs, condition_cache=cache)
        assert prefill.call_count == int(sampling_bf16)
    for got, expected in zip(actual, reference):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    sum(value.square().mean() for value in actual).backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
        else:
            assert parameter.grad is None
