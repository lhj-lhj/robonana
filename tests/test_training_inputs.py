import torch

from robonana.training.robotwin_trainer import flow_noise, image_position_ids, text_position_ids


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
