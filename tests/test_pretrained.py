import torch
from safetensors.torch import save_file

from flux2.model import Flux2, Flux2Params

from robonana.models.pretrained import (
    configure_trainable_parameters,
    load_flux2_fact_checkpoint,
    robot_parameter_names,
)


def _tiny_params():
    return Flux2Params(
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


def test_official_flux_checkpoint_loads_and_only_robot_modules_are_new(tmp_path):
    torch.manual_seed(0)
    base = Flux2(_tiny_params())
    checkpoint = tmp_path / "tiny-flux.safetensors"
    save_file(base.state_dict(), checkpoint)

    model, report = load_flux2_fact_checkpoint(
        checkpoint,
        action_dim=6,
        state_dim=6,
        device="cpu",
        dtype=torch.float32,
        params=_tiny_params(),
    )

    assert torch.equal(model.img_in.weight, base.img_in.weight)
    assert set(report.initialized_robot_parameters) == set(robot_parameter_names(model))
    assert not any(parameter.is_meta for parameter in model.parameters())

    trainable = configure_trainable_parameters(model, "adapters")
    assert set(trainable) == set(robot_parameter_names(model))
    assert not model.double_blocks[0].img_attn.qkv.weight.requires_grad
    configure_trainable_parameters(model, "full")
    assert all(parameter.requires_grad for parameter in model.parameters())
