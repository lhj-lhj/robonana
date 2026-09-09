"""中文：部分重计算须保持前向及梯度；English: test output/gradient parity."""

import pytest
import torch
from unittest.mock import patch
from torch.utils.checkpoint import checkpoint

from test_mac_prefix_cache import model_and_inputs


def test_partial_checkpointing_preserves_world_outputs_and_gradients():
    model, inputs = model_and_inputs()
    model.train()
    model.set_training_phase("world_policy")
    args = dict(inputs, noisy_future_latents=torch.randn(2, 2, 8),
                future_ids=inputs["current_ids"], noisy_pred_action=torch.randn(2, 48, 6),
                gt_action_cond=torch.randn(2, 48, 6), chunk_horizon=torch.full((2,), 48),
                noisy_future_state=torch.randn(2, 1, 6),
                noisy_reward=torch.empty(2, 0, 1), noisy_q=torch.empty(2, 0, 1),
                action_timestep=torch.full((2,), 0.4), wm_timestep=torch.full((2,), 0.6))
    reference = None
    reference_grads = None
    for enabled, stride, calls in [(False, 1, 0), (True, 1, 4), (True, 2, 3)]:
        model.gradient_checkpointing = enabled
        model.set_gradient_checkpointing_single_stride(stride)
        model.zero_grad(set_to_none=True)
        with patch("robonana.models.mac_flux2_fact.checkpoint", wraps=checkpoint) as wrapped:
            output = model(**args)
            fields = [output.image, output.action, output.future_state, output.reward, output.success]
            sum(x.square().mean() for x in fields).backward()
            assert wrapped.call_count == calls
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        if reference is None:
            reference = [x.detach().clone() for x in fields]
            reference_grads = grads
        else:
            for actual, expected in zip(fields, reference):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert grads.keys() == reference_grads.keys()
            for name in grads:
                torch.testing.assert_close(grads[name], reference_grads[name], rtol=1e-5, atol=1e-6)
    model.eval()
    with patch("robonana.models.mac_flux2_fact.checkpoint", side_effect=AssertionError("eval recomputed")):
        model(**args)
    with pytest.raises(ValueError, match="positive integer"):
        model.set_gradient_checkpointing_single_stride(0)
