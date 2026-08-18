import torch

from robonana.sampling import flow_euler_schedule, flow_euler_step


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
