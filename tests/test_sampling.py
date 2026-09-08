from types import SimpleNamespace

import torch
import pytest

from robonana.sampling import (
    flow_euler_schedule,
    flow_euler_step,
    generate_mac_imaginary_rollout_h1,
    sample_q_rejection,
    sample_action_flow,
    sample_flux2_action,
)


class _FakeMacModel:
    architecture_version = "mac_mot_v2"
    chunk_horizon = 48
    action_dim = 2

    def __init__(self, value: float = 0.0):
        self.value = float(value)
        self.prefill_count = 0

    def prefill_condition_cache(self, **kwargs):
        self.prefill_count += 1
        return kwargs["context"]

    def predict_action_cached(self, cache, action, **kwargs):
        return torch.zeros_like(action)

    def condition_cache_compatible(self, cache):
        return cache is not None

    def prefill_world_cache(self, *, clean_action, **kwargs):
        return SimpleNamespace(reward=clean_action.new_zeros(clean_action.shape[0], 48),
                               success=clean_action.new_full((clean_action.shape[0], 1), -20.0))

    def predict_world_cached(self, cache, *, noisy_future_latents, noisy_future_state, **kwargs):
        return SimpleNamespace(image=torch.zeros_like(noisy_future_latents),
                               future_state=torch.zeros_like(noisy_future_state),
                               reward=cache.reward, success=cache.success)

    def score_q_candidates(self, cache, clean_actions, **kwargs):
        return clean_actions.float().mean(dim=(2, 3))

    def __call__(self, **kwargs):
        action = kwargs["noisy_pred_action"]
        clean_action = kwargs["gt_action_cond"]
        batch = kwargs["context"].shape[0]
        device = kwargs["context"].device
        dtype = kwargs["context"].dtype
        q = (
            clean_action.float().mean(dim=(1, 2), keepdim=False)[:, None]
            if clean_action.shape[1]
            else torch.zeros(batch, 1, device=device)
        )
        value = torch.full((batch, 1), self.value, device=device, dtype=dtype)
        if kwargs.get("critic_kind") == "both":
            return value, q.to(device=device, dtype=dtype)
        return SimpleNamespace(
            action=torch.zeros_like(action),
            image=torch.zeros_like(kwargs["noisy_future_latents"]),
            future_state=torch.zeros_like(kwargs["noisy_future_state"]),
            reward=torch.zeros(batch, 48, device=device, dtype=dtype),
            success=torch.full((batch, 1), -20.0, device=device, dtype=dtype),
            q=None,
            value=None,
        )

    def predict_value(self, *, context, expert=None, **kwargs):
        value = self.value if expert is None else expert.value
        return torch.full(
            (context.shape[0], 1), value, device=context.device, dtype=context.dtype
        )


def test_q_rejection_returns_argmax_candidate():
    model = _FakeMacModel()
    context = torch.zeros(1, 1, 3)
    current = torch.zeros(1, 1, 4)
    state = torch.zeros(1, 1, 2)
    noise = torch.stack(
        [
            torch.zeros(48, 2),
            torch.ones(48, 2),
            -torch.ones(48, 2),
        ],
        dim=0,
    )[None]
    result = sample_q_rejection(
        model=model,
        context=context,
        current_latents=current,
        state=state,
        context_mask=torch.ones(1, 1, dtype=torch.bool),
        candidate_count=3,
        action_noise=noise,
        schedule=torch.tensor([1.0, 0.0]),
        grid_height=1,
        grid_width=1,
    )
    assert result.best_index.item() == 1
    assert model.prefill_count == 1
    torch.testing.assert_close(result.action, torch.ones(1, 48, 2))


def test_h1_imaginary_target_uses_binary_reward_curve_and_ema_value():
    online = _FakeMacModel()
    target_value = SimpleNamespace(value=0.1)
    discount = 0.9
    rollout = generate_mac_imaginary_rollout_h1(
        online_model=online,
        target_value_expert=target_value,
        context=torch.zeros(1, 1, 3),
        current_latents=torch.zeros(1, 1, 4),
        state=torch.zeros(1, 1, 2),
        context_mask=torch.ones(1, 1, dtype=torch.bool),
        candidate_count=1,
        action_noise=torch.zeros(1, 1, 48, 2),
        future_noise=torch.zeros(1, 1, 4),
        future_state_noise=torch.zeros(1, 1, 2),
        schedule=torch.tensor([1.0, 0.0]),
        discount=discount,
        reward_non_goal=-1.0,
        reward_goal=0.0,
        return_scale=1000.0,
        grid_height=1,
        grid_width=1,
    )
    expected_reward = -0.5 * sum(discount**step for step in range(48))
    expected_value = expected_reward + discount**48 * 100.0
    torch.testing.assert_close(
        rollout.value_target_return,
        torch.tensor([[expected_value]]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        rollout.q_target_return,
        torch.tensor([[expected_reward]]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert not rollout.value_target_return.requires_grad
    assert not rollout.q_target_return.requires_grad
    assert online.prefill_count == 2  # One current C, one next C for both Values.


def test_flow_euler_schedule_runs_from_pure_noise_to_clean():
    schedule = flow_euler_schedule(4, flow_shift=1.0, device="cpu")
    torch.testing.assert_close(schedule, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))


def test_exact_constant_velocity_recovers_clean_sample():
    clean = torch.tensor([1.0, 2.0])
    noise = torch.tensor([5.0, 8.0])
    velocity = noise - clean
    sample = noise.clone()
    schedule = flow_euler_schedule(5, flow_shift=1.0, device="cpu")
    for sigma, sigma_next in zip(schedule[:-1], schedule[1:]):
        sample = flow_euler_step(sample, velocity, sigma, sigma_next)
    torch.testing.assert_close(sample, clean)


def test_action_only_sampler_uses_the_shared_schedule():
    clean = torch.tensor([[1.0, 2.0]])
    noise = torch.tensor([[5.0, 8.0]])
    result = sample_action_flow(
        action_noise=noise,
        schedule=flow_euler_schedule(5, flow_shift=1.0, device="cpu"),
        predict_action=lambda sample, sigma: noise - clean,
    )
    torch.testing.assert_close(result, clean)


@pytest.mark.parametrize("architecture,horizon", [("mac_mot_v2", 48), ("legacy", 48), ("mac_mot_v2", 24)])
def test_action_sampler_is_mac_only_and_uses_one_condition_cache(architecture, horizon):
    model = _FakeMacModel()
    model.architecture_version = architecture
    noise = torch.randn(2, 48, 2)
    kwargs = dict(model=model, context=torch.zeros(2, 1, 3),
                  current_latents=torch.zeros(2, 1, 4), state=torch.zeros(2, 1, 2),
                  context_mask=torch.ones(2, 1, dtype=torch.bool), action_noise=noise,
                  chunk_horizon=horizon, schedule=flow_euler_schedule(3, flow_shift=1.0, device="cpu"),
                  grid_height=1, grid_width=1)
    if architecture != "mac_mot_v2" or horizon != 48:
        with pytest.raises(ValueError, match="mac_mot_v2"):
            sample_flux2_action(**kwargs)
        assert model.prefill_count == 0
    else:
        torch.testing.assert_close(sample_flux2_action(**kwargs), noise)
        assert model.prefill_count == 1
