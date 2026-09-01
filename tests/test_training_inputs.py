import inspect
from types import SimpleNamespace

import torch
from flux2.model import Flux2Params

from robonana.inference.robotwin_policy import (
    InferenceMode,
    observation_component_digests,
    observation_digest,
    postprocess_action,
    preprocess_action_chunk,
    seeded_randn_like,
)
from robonana.models.flux2_fact import Flux2FACTModel
from robonana.sampling import flow_euler_schedule
from robonana.models.position_ids import dino_position_ids
from robonana.training.robotwin_trainer import flow_noise, image_position_ids, text_position_ids
from robonana.training.robotwin_trainer import RoboNanaTrainer
from world_action_model.pipeline.utils import NormalizationTensors


def test_flow_noise_target_reconstructs_clean_sample():
    torch.manual_seed(7)
    clean = torch.randn(2, 3, 4)
    timestep = torch.tensor([0.25, 0.75])
    noisy, target = flow_noise(clean, timestep)
    restored = noisy - target * timestep[:, None, None]
    torch.testing.assert_close(restored, clean)


def test_trainer_batch_contract_uses_reward_and_q_not_legacy_value():
    source = inspect.getsource(RoboNanaTrainer.forward_step)
    assert 'batch_dict["reward"]' in source
    assert 'batch_dict["success"]' in source
    assert "flow_noise(reward" not in source
    assert 'batch_dict["q"]' in source
    assert 'batch_dict["value"]' not in source


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
    dino_ids = dino_position_ids(
        2,
        num_cameras=3,
        grid_height=2,
        grid_width=2,
        time_coord=torch.tensor([3, 5]),
        device=torch.device("cpu"),
    )
    assert dino_ids.shape == (2, 12, 4)
    assert dino_ids[0, :, 0].unique().item() == 3
    assert dino_ids[0, :, 1:].tolist() == [
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        [2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1],
    ]


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


def test_external_action_preprocess_inverts_postprocess_without_clipping():
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
    state = torch.tensor([0.5, 0.5])
    normalized = torch.tensor([[0.25, 0.25], [-0.5, 0.1]])
    absolute = postprocess_action(
        normalized,
        state,
        normalization,
        delta_mask=torch.tensor([True, False]),
    )
    restored = preprocess_action_chunk(
        absolute,
        state,
        normalization,
        delta_mask=torch.tensor([True, False]),
    )
    torch.testing.assert_close(restored, normalized)


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


def test_online_policy_returns_raw_chunk_reward_q_contract(monkeypatch):
    from robonana.inference.robotwin_policy import RoboNanaRobotWinPolicy

    zeros = torch.zeros(2)
    ones = torch.ones(2)
    policy = object.__new__(RoboNanaRobotWinPolicy)
    policy.model_device = torch.device("cpu")
    policy.vae_device = torch.device("cpu")
    policy.dtype = torch.float32
    policy.state_dim = 2
    policy.horizon = 24
    policy.inference_mode = InferenceMode.ACTION
    policy.return_chunk_q = True
    policy.return_stage2_image = True
    policy.delta_mask = torch.tensor([False, False])
    policy.model = SimpleNamespace(reward_dim=1, q_dim=1)
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
            reward=torch.tensor([[[-3.0]]]),
            success=torch.tensor([[[-2.0]]]),
            q=torch.tensor([[[-7.0]]]),
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
    assert response["chunk_reward"] == -3.0
    assert response["chunk_q"] == -7.0
    assert response["return_horizon"] == 24
    torch.testing.assert_close(response["rewards"], torch.tensor([-3.0]))
    torch.testing.assert_close(response["qs"], torch.tensor([-7.0]))
    assert response["selected_index"] == 0
    assert response["images"].shape == (1, 3, 1, 4, 8)


def test_online_reward_q_sampler_runs_full_world_path_reproducibly():
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
    policy.action_dim = 4
    policy.max_horizon = 4
    policy.stage2_image_horizon_batch_size = 2
    policy.model = Flux2FACTModel(
        params,
        action_dim=4,
        state_dim=3,
        reward_dim=1,
        q_dim=1,
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
    assert first.reward.shape == (1, 1, 1)
    assert first.success.shape == (1, 1, 1)
    assert first.q.shape == (1, 1, 1)
    assert torch.isfinite(first.future).all()
    assert torch.isfinite(first.reward).all()
    assert torch.isfinite(first.success).all()
    assert torch.isfinite(first.q).all()
    torch.testing.assert_close(first.future, second.future)
    torch.testing.assert_close(first.reward, second.reward)
    torch.testing.assert_close(first.success, second.success)
    torch.testing.assert_close(first.q, second.q)


def test_packed_stage2_samples_all_rewards_and_qs_without_future_image_tokens():
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
    policy.grid_height = 1
    policy.grid_width = 2
    policy.state_dim = 3
    policy.action_dim = 4
    policy.max_horizon = 4
    policy.model = Flux2FACTModel(
        params,
        action_dim=4,
        state_dim=3,
        reward_dim=1,
        q_dim=1,
        max_horizon=4,
    ).eval()
    policy.schedule = flow_euler_schedule(1, flow_shift=1.0, device="cpu")
    result = policy._sample_stage2(
        context=torch.randn(1, 2, 16),
        current=torch.randn(1, 2, 8),
        state=torch.randn(1, 1, 3),
        clean_action=torch.randn(1, 4, 4),
        horizons=torch.tensor([1, 2, 3, 4]),
        include_image=False,
        sampling_seed=31,
    )

    assert result.future.shape == (1, 4, 0, 8)
    assert result.future_state.shape == (1, 4, 3)
    assert result.reward.shape == (1, 4, 1)
    assert result.success.shape == (1, 4, 1)
    assert result.q.shape == (1, 4, 1)
    assert torch.isfinite(result.future_state).all()
    assert torch.isfinite(result.reward).all()
    assert torch.isfinite(result.success).all()
    assert torch.isfinite(result.q).all()


def test_image_stage2_chunks_horizons_and_restores_order():
    from robonana.inference.robotwin_policy import RoboNanaRobotWinPolicy

    policy = object.__new__(RoboNanaRobotWinPolicy)
    policy.model_device = torch.device("cpu")
    policy.stage2_image_horizon_batch_size = 2
    calls = []

    def sample_chunk(**kwargs):
        horizons = kwargs["horizons"]
        calls.append(horizons.tolist())
        count = horizons.numel()
        return SimpleNamespace(
            future=horizons.float().reshape(1, count, 1, 1),
            future_state=horizons.float().reshape(1, count, 1),
            reward=horizons.float().reshape(1, count, 1),
            success=torch.zeros(1, count, 1),
            q=-horizons.float().reshape(1, count, 1),
        )

    policy._sample_stage2_chunk = sample_chunk
    result = policy._sample_stage2(
        context=torch.empty(1, 0, 1),
        current=torch.empty(1, 0, 1),
        state=torch.empty(1, 0, 1),
        clean_action=torch.empty(1, 0, 1),
        horizons=torch.arange(1, 6),
        include_image=True,
        sampling_seed=7,
    )

    assert calls == [[1, 2], [3, 4], [5]]
    assert result.reward.reshape(-1).tolist() == [1, 2, 3, 4, 5]
    assert result.success.shape == (1, 5, 1)
    assert result.q.reshape(-1).tolist() == [-1, -2, -3, -4, -5]


def _mock_policy(mode: InferenceMode):
    from robonana.inference.robotwin_policy import RoboNanaRobotWinPolicy

    policy = object.__new__(RoboNanaRobotWinPolicy)
    policy.model_device = torch.device("cpu")
    policy.vae_device = torch.device("cpu")
    policy.dtype = torch.float32
    policy.state_dim = 2
    policy.action_dim = 2
    policy.action_chunk = 3
    policy.max_horizon = 3
    policy.horizon = 2
    policy.inference_mode = mode
    policy.return_chunk_q = False
    policy.return_stage2_image = False
    policy.discount = 1.0
    policy.reward_non_goal = -1.0
    policy.success_threshold = 0.5
    policy.delta_mask = torch.tensor([False, False])
    policy.model = SimpleNamespace(reward_dim=1, q_dim=1)
    zeros = torch.zeros(2)
    ones = torch.ones(2)
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
    policy._sync = lambda device: None
    policy._current_image_tokens = lambda observation: torch.zeros(1, 2, 8)
    policy._context = lambda instruction: torch.zeros(1, 2, 4)
    return policy


def test_action_reward_q_short_circuits_when_h48_is_not_terminal():
    policy = _mock_policy(InferenceMode.ACTION_REWARD_Q)
    policy._sample_action = lambda **kwargs: torch.zeros(1, 3, 2)
    calls = []

    def sample_stage2(**kwargs):
        calls.append(kwargs["horizons"].tolist())
        return SimpleNamespace(
            future=torch.empty(1, 1, 0, 8),
            future_state=torch.zeros(1, 1, 2),
            reward=torch.tensor([[[-1.0]]]),
            success=torch.tensor([[[-10.0]]]),
            q=torch.tensor([[[-4.0]]]),
        )

    policy._sample_stage2 = sample_stage2
    response = policy.inference(
        {"observation.state": torch.zeros(2), "instruction": "move the object"}
    )

    assert calls == [[3]]
    assert response["horizons"].tolist() == [3]
    torch.testing.assert_close(response["reward_curve"], torch.tensor([-1.0, -1.0, -1.0]))
    assert response["accumulated_reward"] == -3.0
    assert response["terminal_horizon"] is None
    assert response["reward_curve_evaluated"] is False
    torch.testing.assert_close(response["qs"], torch.tensor([-4.0]))
    assert response["chunk_reward"] == -1.0
    assert response["chunk_q"] == -4.0
    assert "future_latents" not in response
    assert "images" not in response


def test_action_reward_q_expands_curve_and_finds_first_terminal():
    policy = _mock_policy(InferenceMode.ACTION_REWARD_Q)
    policy._sample_action = lambda **kwargs: torch.zeros(1, 3, 2)
    calls = []

    def sample_stage2(**kwargs):
        horizons = kwargs["horizons"].tolist()
        calls.append(horizons)
        count = len(horizons)
        if horizons == [3]:
            success = torch.tensor([[[10.0]]])
            reward = torch.tensor([[[0.0]]])
        else:
            success = torch.tensor([[[-10.0], [10.0]]])
            reward = torch.tensor([[[-1.0], [0.0]]])
        return SimpleNamespace(
            future=torch.empty(1, count, 0, 8),
            future_state=torch.zeros(1, count, 2),
            reward=reward,
            success=success,
            q=torch.full((1, count, 1), -4.0),
        )

    policy._sample_stage2 = sample_stage2
    response = policy.inference(
        {"observation.state": torch.zeros(2), "instruction": "move the object"}
    )

    assert calls == [[3], [1, 2]]
    assert response["horizons"].tolist() == [1, 2, 3]
    assert response["terminal_horizon"] == 2
    assert response["accumulated_reward"] == -1.0
    assert response["reward_curve_evaluated"] is True


def test_world_horizon_mode_requires_and_uses_external_action_and_horizon():
    policy = _mock_policy(InferenceMode.WORLD_HORIZON)
    policy._sample_action = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("Stage-1 must not run in world_horizon mode")
    )
    captured = {}

    def sample_stage2(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            future=torch.zeros(1, 1, 2, 8),
            future_state=torch.zeros(1, 1, 2),
            reward=torch.zeros(1, 1, 1),
            success=torch.zeros(1, 1, 1),
            q=torch.zeros(1, 1, 1),
        )

    policy._sample_stage2 = sample_stage2
    policy._decode_stage2_images = lambda future: torch.zeros(1, 3, 1, 4, 8)
    response = policy.inference(
        {
            "observation.state": torch.zeros(2),
            "instruction": "move the object",
            "action_chunk": torch.full((3, 2), 0.25),
            "horizon": 3,
        }
    )

    assert captured["include_image"] is True
    assert captured["horizons"].tolist() == [3]
    torch.testing.assert_close(captured["clean_action"], torch.full((1, 3, 2), 0.25))
    assert response["horizons"].tolist() == [3]
    assert response["future_latents"].shape == (1, 2, 8)
    assert response["images"].shape == (1, 3, 1, 4, 8)


def test_world_all_can_skip_future_image_tokens_and_decode():
    policy = _mock_policy(InferenceMode.WORLD_ALL)
    captured = {}

    def sample_stage2(**kwargs):
        captured.update(kwargs)
        count = len(kwargs["horizons"])
        return SimpleNamespace(
            future=torch.empty(1, count, 0, 8),
            future_state=torch.zeros(1, count, 2),
            reward=torch.zeros(1, count, 1),
            success=torch.zeros(1, count, 1),
            q=torch.zeros(1, count, 1),
        )

    policy._sample_stage2 = sample_stage2
    policy._decode_stage2_images = lambda future: (_ for _ in ()).throw(
        AssertionError("image-free WORLD_ALL must not decode")
    )
    response = policy.inference(
        {
            "observation.state": torch.zeros(2),
            "instruction": "move the object",
            "action_chunk": torch.full((3, 2), 0.25),
            "include_image": False,
        }
    )

    assert captured["include_image"] is False
    assert response["horizons"].tolist() == [1, 2, 3]
    assert "future_latents" not in response
    assert "images" not in response
