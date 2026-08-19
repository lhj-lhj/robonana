import torch

from robonana.sampling import (
    flow_euler_schedule,
    flow_euler_step,
    sample_action_flow,
    sample_two_stage_flow,
)


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
    clean_value = torch.tensor([[6.0]])
    action_noise = torch.tensor([[7.0, 8.0]])
    future_noise = torch.tensor([[9.0, 10.0]])
    state_noise = torch.tensor([[11.0]])
    value_noise = torch.tensor([[12.0]])
    world_actions = []

    def predict_action(sample, sigma):
        return action_noise - clean_action

    def predict_world(future, state, value, action, sigma):
        world_actions.append(action.clone())
        return future_noise - clean_future, state_noise - clean_state, value_noise - clean_value

    result = sample_two_stage_flow(
        action_noise=action_noise,
        future_noise=future_noise,
        future_state_noise=state_noise,
        value_noise=value_noise,
        schedule=schedule,
        predict_action=predict_action,
        predict_world=predict_world,
    )
    torch.testing.assert_close(result.action, clean_action)
    torch.testing.assert_close(result.future, clean_future)
    torch.testing.assert_close(result.future_state, clean_state)
    torch.testing.assert_close(result.value, clean_value)
    assert len(world_actions) == 4
    assert all(torch.equal(action, clean_action) for action in world_actions)
