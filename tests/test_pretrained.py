import torch
import pytest

from flux2.model import Flux2Params
from robonana.models.mac_flux2_fact import MacFlux2FACTModel
from robonana.models.pretrained import (
    configure_trainable_parameters,
    load_flux2_fact_trained_checkpoint,
    robot_parameter_names,
)


def tiny_params():
    return Flux2Params(in_channels=8, context_in_dim=16, hidden_size=32,
                       num_heads=4, depth=1, depth_single_blocks=1,
                       axes_dim=[2, 2, 2, 2], mlp_ratio=2.0,
                       use_guidance_embed=False)


def tiny_model():
    return MacFlux2FACTModel(tiny_params(), action_dim=6, state_dim=5,
                             expert_hidden_dim=32)


def test_trained_mac_checkpoint_loads_exactly(tmp_path):
    torch.manual_seed(11)
    expected = tiny_model()
    checkpoint = tmp_path / "mac.bin"
    torch.save(expected.state_dict(), checkpoint)
    actual, report = load_flux2_fact_trained_checkpoint(
        checkpoint, action_dim=6, state_dim=5, reward_dim=48, success_dim=1,
        q_dim=1, reward_head_type="binary_chunk", max_horizon=48,
        chunk_horizon=48, value_dim=1, expert_hidden_dim=32,
        params=tiny_params(), device="cpu", dtype=torch.float32,
    )
    assert actual.architecture_version == "mac_mot_v2"
    assert report.initialized_robot_parameters == ()
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value)


def test_trainable_surfaces_are_only_world_or_critic():
    model = tiny_model()
    world = configure_trainable_parameters(model, "world_policy")
    assert not any(name.startswith(("value_expert.", "q_expert.")) for name in world)
    critic = configure_trainable_parameters(model, "critic")
    assert set(critic) == set(name for name in robot_parameter_names(model)
                               if name.startswith(("value_expert.", "q_expert.")))
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters()
               if not name.startswith(("value_expert.", "q_expert.")))


def test_legacy_schema_is_rejected(tmp_path):
    checkpoint = tmp_path / "legacy.bin"
    torch.save({}, checkpoint)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_flux2_fact_trained_checkpoint(checkpoint, device="cpu")
