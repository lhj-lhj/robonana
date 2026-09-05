from types import SimpleNamespace

import torch

from robonana.sampling import (
    flow_euler_schedule,
    flow_euler_step,
    generate_mac_imaginary_rollout_h1,
    sample_q_rejection,
    sample_action_flow,
    sample_two_stage_flow,
    sample_world_flow,
)


class _FakeMacModel:
    architecture_version = "mac_v1"
    chunk_horizon = 48
    action_dim = 2

    def __init__(self, value: float = 0.0):
        self.value = float(value)

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
        return SimpleNamespace(
            action=torch.zeros_like(action),
            image=torch.zeros_like(kwargs["noisy_future_latents"]),
            future_state=torch.zeros_like(kwargs["noisy_future_state"]),
            reward=torch.zeros(batch, 48, device=device, dtype=dtype),
            success=torch.zeros(batch, 1, device=device, dtype=dtype),
            q=q.to(device=device, dtype=dtype),
            value=torch.full((batch, 1), self.value, device=device, dtype=dtype),
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
    torch.testing.assert_close(result.action, torch.ones(1, 48, 2))


def test_h1_imaginary_target_uses_binary_reward_curve_and_ema_value():
    online = _FakeMacModel()
    ema = _FakeMacModel(value=0.1)
    discount = 0.9
    rollout = generate_mac_imaginary_rollout_h1(
        online_model=online,
        ema_model=ema,
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
    expected = expected_reward + discount**48 * 0.5 * 100.0
    torch.testing.assert_close(
        rollout.target_return, torch.tensor([[expected]]), atol=1e-5, rtol=1e-5
    )
    assert not rollout.target_return.requires_grad


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


def test_two_stage_sampler_feeds_clean_action_into_world_stage():
    schedule = flow_euler_schedule(4, flow_shift=1.0, device="cpu")
    clean_action = torch.tensor([[1.0, 2.0]])
    clean_future = torch.tensor([[3.0, 4.0]])
    clean_state = torch.tensor([[5.0]])
    clean_reward = torch.tensor([[6.0]])
    clean_success = torch.tensor([[8.0]])
    clean_q = torch.tensor([[7.0]])
    action_noise = torch.tensor([[7.0, 8.0]])
    future_noise = torch.tensor([[9.0, 10.0]])
    state_noise = torch.tensor([[11.0]])
    reward_template = torch.zeros_like(clean_reward)
    q_noise = torch.tensor([[13.0]])
    world_actions = []

    def predict_action(sample, sigma):
        return action_noise - clean_action

    def predict_world(future, state, reward_query, q, action, sigma):
        world_actions.append(action.clone())
        return (
            future_noise - clean_future,
            state_noise - clean_state,
            clean_reward,
            clean_success,
            q_noise - clean_q,
        )

    result = sample_two_stage_flow(
        action_noise=action_noise,
        future_noise=future_noise,
        future_state_noise=state_noise,
        reward_template=reward_template,
        q_noise=q_noise,
        schedule=schedule,
        predict_action=predict_action,
        predict_world=predict_world,
    )
    torch.testing.assert_close(result.action, clean_action)
    torch.testing.assert_close(result.future, clean_future)
    torch.testing.assert_close(result.future_state, clean_state)
    torch.testing.assert_close(result.reward, clean_reward)
    torch.testing.assert_close(result.success, clean_success)
    torch.testing.assert_close(result.q, clean_q)
    assert len(world_actions) == 5
    assert all(torch.equal(action, clean_action) for action in world_actions)


def test_world_sampler_uses_supplied_gt_action_without_action_sampling():
    schedule = flow_euler_schedule(4, flow_shift=1.0, device="cpu")
    clean_action = torch.tensor([[1.0, 2.0]])
    clean_future = torch.tensor([[3.0, 4.0]])
    clean_state = torch.tensor([[5.0]])
    clean_reward = torch.tensor([[6.0]])
    clean_success = torch.tensor([[8.0]])
    clean_q = torch.tensor([[7.0]])
    future_noise = torch.tensor([[9.0, 10.0]])
    state_noise = torch.tensor([[11.0]])
    reward_template = torch.zeros_like(clean_reward)
    q_noise = torch.tensor([[13.0]])
    world_actions = []

    def predict_world(future, state, reward_query, q, action, sigma):
        world_actions.append(action.clone())
        return (
            future_noise - clean_future,
            state_noise - clean_state,
            clean_reward,
            clean_success,
            q_noise - clean_q,
        )

    result = sample_world_flow(
        clean_action=clean_action,
        future_noise=future_noise,
        future_state_noise=state_noise,
        reward_template=reward_template,
        q_noise=q_noise,
        schedule=schedule,
        predict_world=predict_world,
    )
    torch.testing.assert_close(result.future, clean_future)
    torch.testing.assert_close(result.future_state, clean_state)
    torch.testing.assert_close(result.reward, clean_reward)
    torch.testing.assert_close(result.success, clean_success)
    torch.testing.assert_close(result.q, clean_q)
    assert len(world_actions) == 5
    assert all(torch.equal(action, clean_action) for action in world_actions)
