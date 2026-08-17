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
        noisy_value=torch.randn(batch, 1, 1),
        action_timestep=torch.rand(batch),
        wm_timestep=torch.rand(batch),
        context_mask=torch.ones(batch, context_len, dtype=torch.bool),
    )

    assert output.image.shape == (batch, image_len, 8)
    assert output.action.shape == (batch, action_len, 6)
    assert output.future_state.shape == (batch, 1, 6)
    assert output.value.shape == (batch, 1, 1)

    loss = output.image.square().mean() + output.action.square().mean()
    loss.backward()
    assert model.double_blocks[0].img_attn.qkv.weight.grad is not None
    assert model.single_blocks[0].linear1.weight.grad is not None

