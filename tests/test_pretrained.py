import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

from flux2.model import Flux2, Flux2Params

from robonana.models.flux2_fact import Flux2FACTModel
from robonana.models.pretrained import (
    configure_trainable_parameters,
    load_flux2_fact_checkpoint,
    load_flux2_fact_trained_checkpoint,
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


def test_full_trained_checkpoint_loads_exactly_from_fact_export(tmp_path):
    torch.manual_seed(11)
    expected = Flux2FACTModel(
        _tiny_params(),
        action_dim=6,
        state_dim=6,
        value_dim=1,
        max_horizon=8,
    )
    checkpoint = tmp_path / "diffusion_pytorch_model.bin"
    torch.save(expected.state_dict(), checkpoint)

    actual, report = load_flux2_fact_trained_checkpoint(
        checkpoint,
        action_dim=6,
        state_dim=6,
        value_dim=1,
        max_horizon=8,
        device="cpu",
        dtype=torch.float32,
        params=_tiny_params(),
    )

    assert report.initialized_robot_parameters == ()
    assert report.checkpoint_parameters == sum(p.numel() for p in expected.parameters())
    for name, expected_tensor in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], expected_tensor)


def test_standalone_trained_checkpoint_requires_exact_model_config(tmp_path):
    expected = Flux2FACTModel(
        _tiny_params(),
        action_dim=6,
        state_dim=5,
        value_dim=2,
        max_horizon=8,
    )
    checkpoint = tmp_path / "standalone.bin"
    torch.save(expected.state_dict(), checkpoint)

    with pytest.raises(FileNotFoundError, match="pass --model-config explicitly"):
        load_flux2_fact_trained_checkpoint(
            checkpoint,
            device="cpu",
            dtype=torch.float32,
        )


def test_trained_checkpoint_discovers_fact_project_config(tmp_path):
    project = tmp_path / "experiment"
    transformer = project / "models" / "checkpoint_epoch_1_step_10" / "transformer"
    transformer.mkdir(parents=True)
    checkpoint = transformer / "diffusion_pytorch_model.bin"
    expected = Flux2FACTModel(
        _tiny_params(),
        action_dim=6,
        state_dim=5,
        value_dim=2,
        max_horizon=8,
    )
    torch.save(expected.state_dict(), checkpoint)
    config_path = project / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "params": asdict(_tiny_params()),
                    "action_dim": 6,
                    "state_dim": 5,
                    "value_dim": 2,
                    "max_horizon": 8,
                }
            }
        ),
        encoding="utf-8",
    )

    actual, report = load_flux2_fact_trained_checkpoint(
        checkpoint,
        device="cpu",
        dtype=torch.float32,
    )

    assert actual.hidden_size == _tiny_params().hidden_size
    assert report.model_config.source == str(config_path.resolve())


def test_incomplete_project_config_fails_before_loading_checkpoint(tmp_path):
    checkpoint = tmp_path / "models" / "step_1" / "transformer" / "model.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not loaded")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "params": asdict(_tiny_params()),
                    "action_dim": 6,
                    "state_dim": 5,
                    "max_horizon": 8,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="value_dim"):
        load_flux2_fact_trained_checkpoint(
            checkpoint,
            device="cpu",
            dtype=torch.float32,
        )
