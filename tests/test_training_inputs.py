from types import SimpleNamespace

import torch
from flux2.model import Flux2Params

from robonana.inference.robotwin_policy import (
    observation_component_digests,
    observation_digest,
    postprocess_action,
    seeded_randn_like,
)
from robonana.models.flux2_fact import Flux2FACTModel
from robonana.sampling import flow_euler_schedule
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


def test_observation_digest_tracks_inputs_not_dictionary_identity():
    observation = {
        "observation.state": torch.zeros(14),
        "observation.images.cam_high": torch.zeros(3, 4, 5),
        "observation.images.cam_left_wrist": torch.zeros(3, 4, 5),
        "observation.images.cam_right_wrist": torch.zeros(3, 4, 5),
        "instruction": "beat the block with the hammer",
    }
    copied = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in observation.items()}
    changed = dict(copied)
    changed["observation.state"] = torch.ones(14)

    assert observation_digest(observation) == observation_digest(copied)
    assert observation_digest(observation) != observation_digest(changed)
    assert observation_component_digests(observation) == observation_component_digests(copied)
    assert (
        observation_component_digests(observation)["state"]
        != observation_component_digests(changed)["state"]
    )


def test_online_policy_returns_denormalized_chunk_value_contract(monkeypatch):
    from robonana.inference.robotwin_policy import RoboNanaRobotWinPolicy

    zeros = torch.zeros(2)
    ones = torch.ones(2)
    policy = object.__new__(RoboNanaRobotWinPolicy)
    policy.model_device = torch.device("cpu")
    policy.vae_device = torch.device("cpu")
    policy.dtype = torch.float32
    policy.state_dim = 2
    policy.horizon = 24
    policy.return_chunk_value = True
    policy.return_stage2_image = True
    policy.delta_mask = torch.tensor([False, False])
    policy.model = SimpleNamespace(value_dim=1)
    policy.normalization = NormalizationTensors(
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
    monkeypatch.setattr(policy, "_sync", lambda device: None)
    monkeypatch.setattr(
        policy,
        "_current_image_tokens",
        lambda observation: torch.zeros(1, 6, 8),
    )
    monkeypatch.setattr(policy, "_context", lambda instruction: torch.zeros(1, 2, 4))
    monkeypatch.setattr(
        policy,
        "_sample_action",
        lambda **kwargs: torch.zeros(1, 3, 2),
    )
    monkeypatch.setattr(
        policy,
        "_sample_world",
        lambda **kwargs: SimpleNamespace(
            future=torch.zeros(1, 6, 8),
            future_state=torch.zeros(1, 1, 2),
            value=torch.zeros(1, 1, 1),
        ),
    )
    monkeypatch.setattr(
        policy,
        "_decode_stage2_image",
        lambda future: torch.zeros(1, 3, 1, 4, 8),
    )

    response = policy.inference(
        {
            "observation.state": torch.zeros(2),
            "instruction": "move the object",
        }
    )

    assert response["action"].shape == (3, 2)
    assert response["chunk_value"] == 0.5
    assert response["value_horizon"] == 24
    torch.testing.assert_close(response["values_per_sample"], torch.tensor([0.5]))
    assert response["selected_index"] == 0
    assert response["images"].shape == (1, 3, 1, 4, 8)


def test_online_value_sampler_runs_full_world_path_reproducibly():
    from robonana.inference.robotwin_policy import RoboNanaRobotWinPolicy

    params = Flux2Params(
        in_channels=8,
        context_in_dim=16,
        hidden_size=32,
        num_heads=4,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[2, 2, 2, 2],
        mlp_ratio=2.0,
        use_guidance_embed=False,
    )
    policy = object.__new__(RoboNanaRobotWinPolicy)
    policy.model_device = torch.device("cpu")
    policy.dtype = torch.float32
    policy.horizon = 2
    policy.grid_height = 1
    policy.grid_width = 2
    policy.state_dim = 3
    policy.model = Flux2FACTModel(
        params,
        action_dim=4,
        state_dim=3,
        value_dim=1,
        max_horizon=4,
    ).eval()
    policy.schedule = flow_euler_schedule(1, flow_shift=1.0, device="cpu")
    kwargs = {
        "context": torch.randn(1, 2, 16),
        "current": torch.randn(1, 2, 8),
        "state": torch.randn(1, 1, 3),
        "clean_action": torch.randn(1, 3, 4),
        "sampling_seed": 123,
    }

    first = policy._sample_world(**kwargs)
    second = policy._sample_world(**kwargs)

    assert first.future.shape == (1, 2, 8)
    assert first.value.shape == (1, 1, 1)
    assert torch.isfinite(first.future).all()
    assert torch.isfinite(first.value).all()
    torch.testing.assert_close(first.future, second.future)
    torch.testing.assert_close(first.value, second.value)
