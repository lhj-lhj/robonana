import torch

from robonana.inference.robotwin_policy import postprocess_action, seeded_randn_like
from robonana.training.robotwin_trainer import flow_noise, image_position_ids, text_position_ids
from world_action_model.pipeline.utils import NormalizationTensors


def test_flow_noise_target_reconstructs_clean_sample():
    torch.manual_seed(7)
    clean = torch.randn(2, 3, 4)
    timestep = torch.tensor([0.25, 0.75])
    noisy, target = flow_noise(clean, timestep)
    restored = noisy - target * timestep[:, None, None]
    torch.testing.assert_close(restored, clean)


def test_flux_position_ids_encode_language_space_and_horizon_time():
    text_ids = text_position_ids(2, 3, torch.device("cpu"))
    assert text_ids[0, :, 3].tolist() == [0, 1, 2]
    image_ids = image_position_ids(
        2,
        grid_height=2,
        grid_width=3,
        time_coord=torch.tensor([1, 4]),
        device=torch.device("cpu"),
    )
    assert image_ids.shape == (2, 6, 4)
    assert image_ids[0, :, 0].unique().item() == 1
    assert image_ids[1, :, 0].unique().item() == 4
    assert image_ids[0, :, 1:3].tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]


def test_online_action_postprocess_matches_training_delta_convention():
    zeros = torch.zeros(2)
    ones = torch.ones(2)
    normalization = NormalizationTensors(
        state_mean=zeros,
        state_std=ones,
        state_min=torch.full((2,), -2.0),
        state_max=torch.full((2,), 2.0),
        action_mean=zeros,
        action_std=ones,
        action_min=torch.full((2,), -1.0),
        action_max=torch.full((2,), 1.0),
        value_min=torch.tensor([-1.0]),
        value_max=torch.tensor([2.0]),
    )
    result = postprocess_action(
        torch.full((3, 2), 0.25),
        torch.tensor([0.5, 0.5]),
        normalization,
        delta_mask=torch.tensor([True, False]),
    )
    torch.testing.assert_close(
        result,
        torch.tensor([[0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]),
    )


def test_seeded_action_noise_is_reproducible_without_mutating_global_rng():
    reference = torch.empty(3, 4)
    torch.manual_seed(9)
    before = torch.random.get_rng_state()
    first = seeded_randn_like(reference, 123)
    after = torch.random.get_rng_state()
    second = seeded_randn_like(reference, 123)

    assert torch.equal(before, after)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, seeded_randn_like(reference, 124))
