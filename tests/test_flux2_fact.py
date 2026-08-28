import torch

from flux2.model import Flux2, Flux2Params

from robonana.models.flux2_fact import Flux2FACTModel


def _tiny_model():
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
    return Flux2FACTModel(params, action_dim=6, state_dim=6, max_horizon=8)


def test_model_inherits_flux2_and_runs_shared_blocks():
    torch.manual_seed(0)
    model = _tiny_model()
    assert isinstance(model, Flux2)

    batch, context_len, ref_len, image_len, action_len = 2, 3, 2, 3, 4
    ids = lambda length: torch.zeros(batch, length, 4)
    output = model(
        context=torch.randn(batch, context_len, 16),
        context_ids=ids(context_len),
        current_latents=torch.randn(batch, ref_len, 8),
        current_ids=ids(ref_len),
        noisy_future_latents=torch.randn(batch, image_len, 8),
        future_ids=ids(image_len),
        state=torch.randn(batch, 1, 6),
        noisy_pred_action=torch.randn(batch, action_len, 6),
        gt_action_cond=torch.randn(batch, action_len, 6),
        horizon_idx=torch.tensor([1, 2]),
        noisy_future_state=torch.randn(batch, 1, 6),
        noisy_reward=torch.randn(batch, 1, 1),
        noisy_q=torch.randn(batch, 1, 1),
        action_timestep=torch.rand(batch),
        wm_timestep=torch.rand(batch),
        context_mask=torch.ones(batch, context_len, dtype=torch.bool),
    )

    assert output.image.shape == (batch, image_len, 8)
    assert output.action.shape == (batch, action_len, 6)
    assert output.future_state.shape == (batch, 1, 6)
    assert output.reward.shape == (batch, 1, 1)
    assert output.q.shape == (batch, 1, 1)

    loss = output.image.square().mean() + output.action.square().mean()
    loss.backward()
    assert model.double_blocks[0].img_attn.qkv.weight.grad is not None
    assert model.single_blocks[0].linear1.weight.grad is not None


def test_gradient_checkpointed_shared_blocks_run_backward():
    torch.manual_seed(0)
    model = _tiny_model()
    model.enable_gradient_checkpointing()
    model.train()
    batch = 1
    ids = lambda length: torch.zeros(batch, length, 4)
    output = model(
        context=torch.randn(batch, 2, 16),
        context_ids=ids(2),
        current_latents=torch.randn(batch, 1, 8),
        current_ids=ids(1),
        noisy_future_latents=torch.randn(batch, 1, 8),
        future_ids=ids(1),
        state=torch.randn(batch, 1, 6),
        noisy_pred_action=torch.randn(batch, 2, 6),
        gt_action_cond=torch.randn(batch, 2, 6),
        horizon_idx=torch.tensor([1]),
        noisy_future_state=torch.randn(batch, 1, 6),
        noisy_reward=torch.randn(batch, 1, 1),
        noisy_q=torch.randn(batch, 1, 1),
        action_timestep=torch.rand(batch),
        wm_timestep=torch.rand(batch),
    )
    (
        output.image.square().mean()
        + output.action.square().mean()
        + output.future_state.square().mean()
        + output.reward.square().mean()
        + output.q.square().mean()
    ).backward()
    assert model.double_blocks[0].img_attn.qkv.weight.grad is not None
    assert model.single_blocks[0].linear1.weight.grad is not None
    assert model.reward_out.weight.grad is not None
    assert model.q_out.weight.grad is not None


def test_action_only_forward_accepts_empty_world_suffix():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch, context_len, ref_len, action_len = 1, 2, 2, 4
    ids = lambda length: torch.zeros(batch, length, 4)
    with torch.inference_mode():
        output = model(
            context=torch.randn(batch, context_len, 16),
            context_ids=ids(context_len),
            current_latents=torch.randn(batch, ref_len, 8),
            current_ids=ids(ref_len),
            noisy_future_latents=torch.empty(batch, 0, 8),
            future_ids=ids(0),
            state=torch.randn(batch, 1, 6),
            noisy_pred_action=torch.randn(batch, action_len, 6),
            gt_action_cond=torch.zeros(batch, action_len, 6),
            horizon_idx=torch.tensor([1]),
            noisy_future_state=torch.empty(batch, 0, 6),
            noisy_reward=torch.empty(batch, 0, 1),
            noisy_q=torch.empty(batch, 0, 1),
            action_timestep=torch.rand(batch),
            wm_timestep=torch.zeros(batch),
            context_mask=torch.ones(batch, context_len, dtype=torch.bool),
        )
    assert output.action.shape == (batch, action_len, 6)
    assert output.image.shape == (batch, 0, 8)
    assert output.future_state.shape == (batch, 0, 6)
    assert output.reward.shape == (batch, 0, 1)
    assert output.q.shape == (batch, 0, 1)
    assert output.dino is None


def test_dino_head_is_optional_and_cannot_change_earlier_outputs():
    torch.manual_seed(3)
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
    model = Flux2FACTModel(
        params,
        action_dim=6,
        state_dim=6,
        max_horizon=8,
        dino_dim=12,
    ).eval()
    batch = 1
    ids = lambda length: torch.zeros(batch, length, 4)
    kwargs = dict(
        context=torch.randn(batch, 2, 16),
        context_ids=ids(2),
        current_latents=torch.randn(batch, 2, 8),
        current_ids=ids(2),
        noisy_future_latents=torch.randn(batch, 3, 8),
        future_ids=ids(3),
        state=torch.randn(batch, 1, 6),
        noisy_pred_action=torch.randn(batch, 2, 6),
        gt_action_cond=torch.randn(batch, 2, 6),
        horizon_idx=torch.tensor([2]),
        noisy_future_state=torch.randn(batch, 1, 6),
        noisy_reward=torch.randn(batch, 1, 1),
        noisy_q=torch.randn(batch, 1, 1),
        action_timestep=torch.rand(batch),
        wm_timestep=torch.rand(batch),
        dino_ids=ids(4),
    )
    with torch.inference_mode():
        first = model(noisy_future_dino=torch.zeros(batch, 4, 12), **kwargs)
        second = model(noisy_future_dino=torch.ones(batch, 4, 12), **kwargs)

    assert first.dino.shape == (batch, 4, 12)
    assert not torch.equal(first.dino, second.dino)
    for name in ("image", "action", "future_state", "reward", "q"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name))


def test_packed_horizon_forward_matches_isolated_single_horizon_queries():
    torch.manual_seed(11)
    model = _tiny_model().eval()
    batch, horizon_count, image_tokens = 1, 2, 1
    context = torch.randn(batch, 2, 16)
    current = torch.randn(batch, 2, 8)
    state = torch.randn(batch, 1, 6)
    action = torch.randn(batch, 4, 6)
    horizons = torch.tensor([[1, 3]])
    future_state = torch.randn(batch, horizon_count, 6)
    reward = torch.randn(batch, horizon_count, 1)
    q = torch.randn(batch, horizon_count, 1)
    future = torch.randn(batch, horizon_count, image_tokens, 8)
    context_ids = torch.zeros(batch, 2, 4)
    current_ids = torch.zeros(batch, 2, 4)
    future_ids = torch.zeros(batch, horizon_count, image_tokens, 4)
    common = dict(
        context=context,
        context_ids=context_ids,
        current_latents=current,
        current_ids=current_ids,
        state=state,
        noisy_pred_action=torch.empty(batch, 0, 6),
        gt_action_cond=action,
        action_timestep=torch.zeros(batch),
        wm_timestep=torch.full((batch,), 0.6),
    )
    with torch.inference_mode():
        packed = model(
            noisy_future_latents=future,
            future_ids=future_ids,
            horizon_idx=horizons,
            noisy_future_state=future_state,
            noisy_reward=reward,
            noisy_q=q,
            **common,
        )
        singles = [
            model(
                noisy_future_latents=future[:, index],
                future_ids=future_ids[:, index],
                horizon_idx=horizons[:, index],
                noisy_future_state=future_state[:, index, None],
                noisy_reward=reward[:, index, None],
                noisy_q=q[:, index, None],
                **common,
            )
            for index in range(horizon_count)
        ]

    assert packed.image.shape == (batch, horizon_count, image_tokens, 8)
    assert packed.future_state.shape == (batch, horizon_count, 6)
    assert packed.reward.shape == (batch, horizon_count, 1)
    assert packed.q.shape == (batch, horizon_count, 1)
    torch.testing.assert_close(
        packed.image,
        torch.stack([output.image for output in singles], dim=1),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        packed.future_state,
        torch.cat([output.future_state for output in singles], dim=1),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        packed.reward,
        torch.cat([output.reward for output in singles], dim=1),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        packed.q,
        torch.cat([output.q for output in singles], dim=1),
        atol=2e-5,
        rtol=2e-5,
    )
