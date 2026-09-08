"""Execution must not clip finite actions after Q/world-model evaluation."""

import torch

from robonana.inference.robotwin_policy import postprocess_action
from world_action_model.pipeline.utils import NormalizationTensors


def _normalization():
    return NormalizationTensors(
        state_mean=torch.zeros(2), state_std=torch.ones(2),
        state_min=torch.full((2,), -1.0), state_max=torch.ones(2),
        action_mean=torch.tensor([0.5, -0.5]), action_std=torch.tensor([2.0, 3.0]),
        action_min=torch.full((2,), -1.0), action_max=torch.ones(2),
        value_min=torch.tensor([-1.0]), value_max=torch.tensor([0.0]),
    )


def test_finite_action_exceeding_both_ranges_is_not_clipped():
    stats = _normalization()
    sampled = torch.tensor([[3.0, -4.0], [-3.0, 4.0], [0.1, 0.2]])
    before = sampled.clone()
    state = torch.tensor([10.0, 20.0])
    actual = postprocess_action(sampled, state, stats, delta_mask=torch.tensor([True, False]))
    expected = sampled * stats.action_std + stats.action_mean
    expected[:, 0] += state[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(sampled, before)
    # Invert execution mapping to recover the action originally scored by Q.
    restored = actual.clone()
    restored[:, 0] -= state[0]
    torch.testing.assert_close((restored - stats.action_mean) / stats.action_std, sampled)


def test_nonfinite_action_fallback_is_preserved():
    actual = postprocess_action(
        torch.tensor([[float("nan"), float("inf")], [float("-inf"), 0.0]]),
        torch.tensor([2.0, 3.0]), _normalization(), delta_mask=torch.tensor([True, False]),
    )
    torch.testing.assert_close(actual, torch.tensor([[2.0, 0.0], [2.0, -0.5]]))
