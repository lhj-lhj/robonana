import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

from flux2.model import Flux2, Flux2Params

from robonana.models.flux2_fact import Flux2FACTModel
from robonana.models.flux2_scalar_expert import (
    DeterministicFlux2ScalarExpert,
    initialize_scalar_expert_from_flux,
)
import robonana.models.pretrained as pretrained
from robonana.models.pretrained import (
    configure_trainable_parameters,
    load_flux2_fact_checkpoint,
    load_flux2_fact_trained_checkpoint,
    load_mac_from_legacy_checkpoint,
    robot_parameter_names,
)


def test_mac_migration_from_120k_loads_compatible_weights_only(tmp_path, monkeypatch):
    # Observe the real loader after meta materialization/reset but before
    # FLUX transfer: both experts must retain these exact query/head tensors.
    fresh_experts = []

    def capture_initialization(expert, flux):
        fresh_experts.append({
            name: tensor.clone()
            for name, tensor in expert.state_dict().items()
            if name.startswith(("query.", "head."))
        })
        return initialize_scalar_expert_from_flux(expert, flux)

    monkeypatch.setattr(pretrained, "initialize_scalar_expert_from_flux", capture_initialization)
    source = Flux2FACTModel(
        _tiny_params(), action_dim=6, state_dim=5, max_horizon=48
    )
    checkpoint = tmp_path / "run120k" / "models" / "checkpoint_epoch_6_step_120000" / "transformer" / "diffusion_pytorch_model.bin"
    checkpoint.parent.mkdir(parents=True)
    torch.save(source.state_dict(), checkpoint)
    config_path = tmp_path / "run120k" / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "params": asdict(_tiny_params()),
                    "action_dim": 6,
                    "state_dim": 5,
                    "reward_dim": 1,
                    "success_dim": 1,
                    "q_dim": 1,
                    "reward_head_type": "direct",
                    "max_horizon": 48,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="legacy 120k"):
        model, report = load_mac_from_legacy_checkpoint(
            checkpoint,
            config_path=config_path,
            action_dim=6,
            state_dim=5,
            device="cpu",
            dtype=torch.float32,
            params=_tiny_params(),
        )

    assert model.architecture_version == "mac_mot_v2"
    assert model.reward_out.weight.shape == (48, model.hidden_size)
    torch.testing.assert_close(model.action_in.weight, source.action_in.weight)
    torch.testing.assert_close(model.state_out.weight, source.state_out.weight)
    assert "action_in.weight" in report.loaded_parameter_names
    assert "q_out.weight" in report.skipped_checkpoint_parameters
    assert "horizon_embed.weight" in report.skipped_checkpoint_parameters
    assert any(name.startswith("value_expert.") for name in report.initialized_robot_parameters)
    assert any(name.startswith("q_expert.") for name in report.initialized_robot_parameters)
    assert not hasattr(model, "value_token")
    assert not hasattr(model, "q_token")
    # Tiny test dimensions match exactly, so ImageWAM-style initialization is
    # an exact copy of the loaded FLUX image branch for both experts.
    torch.testing.assert_close(
        model.value_expert.double_blocks[0].img_attn.qkv.weight,
        model.double_blocks[0].img_attn.qkv.weight,
    )
    torch.testing.assert_close(
        model.q_expert.single_blocks[0].linear1.weight,
        model.single_blocks[0].linear1.weight,
    )
    assert len(fresh_experts) == 2
    for expert, fresh in zip((model.value_expert, model.q_expert), fresh_experts):
        for name, expected in fresh.items():
            torch.testing.assert_close(expert.state_dict()[name], expected, rtol=0, atol=0)
    assert not torch.equal(model.value_expert.head.linear.weight, model.q_expert.head.linear.weight)

    # Later rounds must preserve trained task-specific weights; the new
    # initialization boundary must never be applied to exact MAC loading.
    with torch.no_grad():
        for expert in (model.value_expert, model.q_expert):
            expert.query.weight.add_(0.5)
            for parameter in expert.head.parameters():
                parameter.add_(0.5)
    trained_checkpoint = tmp_path / "mac.bin"
    trained_config = tmp_path / "mac_config.json"
    torch.save(model.state_dict(), trained_checkpoint)
    trained_config.write_text(json.dumps(asdict(report.model_config)), encoding="utf-8")
    restored, restored_report = load_flux2_fact_trained_checkpoint(
        trained_checkpoint, config_path=trained_config, device="cpu", dtype=torch.float32
    )
    assert len(fresh_experts) == 2  # Exact loading never invoked the initializer.
    assert restored_report.initialized_robot_parameters == ()
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("hidden_dim", [32, 16])
def test_scalar_expert_transfer_preserves_fresh_query_and_entire_head(hidden_dim):
    source = Flux2(_tiny_params())
    expert = DeterministicFlux2ScalarExpert(
        hidden_dim=hidden_dim, num_heads=4, attn_head_dim=8,
        num_layers_double=1, num_layers_single=1, mlp_ratio=2.0,
    )
    expert.reset_parameters()
    fresh = {name: tensor.clone() for name, tensor in expert.state_dict().items()}
    # Constant body weights give an independently calculable interpolation
    # result. Poison the image head to catch accidental transfer of any part.
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.25)
        for parameter in source.final_layer.parameters():
            parameter.fill_(123.0)
    copied, resized = initialize_scalar_expert_from_flux(expert, source)
    body_count = 0
    for name, actual in expert.state_dict().items():
        if name.startswith(("query.", "head.")):
            torch.testing.assert_close(actual, fresh[name], rtol=0, atol=0)
        else:
            body_count += 1
            source_tensor = source.state_dict()[name]
            scale = 1.0
            if source_tensor.ndim >= 2 and source_tensor.shape[-1] != actual.shape[-1]:
                scale = (source_tensor.shape[-1] / actual.shape[-1]) ** 0.5
            torch.testing.assert_close(actual, torch.full_like(actual, 0.25 * scale))
    assert copied + resized == body_count
    assert (resized > 0) == (hidden_dim != 32)


def test_scalar_expert_transfer_rejects_missing_body_parameter():
    source = Flux2(_tiny_params())
    expert = DeterministicFlux2ScalarExpert(
        hidden_dim=32, num_heads=4, attn_head_dim=8,
        num_layers_double=1, num_layers_single=1, mlp_ratio=2.0,
    )
    del source.time_in
    with pytest.raises(RuntimeError, match="FLUX-to-expert initialization mismatch"):
        initialize_scalar_expert_from_flux(expert, source)


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


def _legacy_value_state_dict(model: Flux2FACTModel):
    state = dict(model.state_dict())
    state.pop("reward_token.weight")
    state["value_in.weight"] = torch.randn(model.hidden_size, 1)
    state["value_out.weight"] = state.pop("reward_out.weight")
    state.pop("success_token.weight")
    state.pop("success_out.weight")
    for prefix in ("q_in.", "q_out.", "q_segment_embed."):
        for name in tuple(state):
            if name.startswith(prefix):
                state.pop(name)
    return state


def _legacy_flow_reward_state_dict(model: Flux2FACTModel):
    state = dict(model.state_dict())
    state.pop("reward_token.weight")
    state.pop("success_token.weight")
    state.pop("success_out.weight")
    state["reward_in.weight"] = torch.randn(model.hidden_size, 1)
    return state


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
        max_horizon=8,
    )
    checkpoint = tmp_path / "diffusion_pytorch_model.bin"
    torch.save(expected.state_dict(), checkpoint)

    actual, report = load_flux2_fact_trained_checkpoint(
        checkpoint,
        action_dim=6,
        state_dim=6,
        reward_dim=1,
        success_dim=1,
        q_dim=1,
        reward_head_type="direct",
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
        max_horizon=8,
        pred_action_bidirectional=True,
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
                    "reward_dim": 1,
                    "success_dim": 1,
                    "q_dim": 1,
                    "reward_head_type": "direct",
                    "max_horizon": 8,
                    "pred_action_bidirectional": True,
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
    assert actual.pred_action_bidirectional is True
    assert report.model_config.pred_action_bidirectional is True
    assert report.model_config.source == str(config_path.resolve())


def test_legacy_project_config_defaults_to_causal_pred_action(tmp_path):
    project = tmp_path / "legacy-experiment"
    transformer = project / "models" / "checkpoint_epoch_1_step_10" / "transformer"
    transformer.mkdir(parents=True)
    checkpoint = transformer / "diffusion_pytorch_model.bin"
    expected = Flux2FACTModel(
        _tiny_params(),
        action_dim=6,
        state_dim=5,
        max_horizon=8,
    )
    torch.save(_legacy_value_state_dict(expected), checkpoint)
    config_path = project / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "params": asdict(_tiny_params()),
                    "action_dim": 6,
                    "state_dim": 5,
                    "value_dim": 1,
                    "max_horizon": 8,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="value_in/value_out"):
        actual, report = load_flux2_fact_trained_checkpoint(
            checkpoint,
            success_dim=1,
            reward_head_type="direct",
            device="cpu",
            dtype=torch.float32,
        )

    assert actual.pred_action_bidirectional is False
    assert report.model_config.pred_action_bidirectional is False
    assert report.model_config.legacy_value_dim == 1
    assert any(name.startswith("reward_token.") for name in report.initialized_robot_parameters)
    assert any(name.startswith("q_in.") for name in report.initialized_robot_parameters)
    new_return_prefixes = (
        "reward_token.",
        "reward_out.",
        "success_token.",
        "success_out.",
        "q_in.",
        "q_out.",
        "q_segment_embed.",
    )
    for name, expected_tensor in expected.state_dict().items():
        if not name.startswith(new_return_prefixes):
            torch.testing.assert_close(actual.state_dict()[name], expected_tensor)


def test_flow_reward_checkpoint_warm_starts_q_but_reinitializes_direct_heads(tmp_path):
    project = tmp_path / "flow-reward"
    transformer = project / "models" / "step_150000" / "transformer"
    transformer.mkdir(parents=True)
    checkpoint = transformer / "diffusion_pytorch_model.bin"
    expected = Flux2FACTModel(_tiny_params(), action_dim=6, state_dim=5, max_horizon=8)
    torch.save(_legacy_flow_reward_state_dict(expected), checkpoint)
    (project / "config.json").write_text(
        json.dumps(
            {
                "models": {
                    "params": asdict(_tiny_params()),
                    "action_dim": 6,
                    "state_dim": 5,
                    "reward_dim": 1,
                    "q_dim": 1,
                    "max_horizon": 8,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="flow-matched reward"):
        actual, report = load_flux2_fact_trained_checkpoint(
            checkpoint,
            success_dim=1,
            reward_head_type="direct",
            device="cpu",
            dtype=torch.float32,
        )

    initialized = set(report.initialized_robot_parameters)
    assert initialized == {
        "reward_out.weight",
        "reward_token.weight",
        "success_out.weight",
        "success_token.weight",
    }
    torch.testing.assert_close(actual.q_in.weight, expected.q_in.weight)
    torch.testing.assert_close(actual.q_out.weight, expected.q_out.weight)


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

    with pytest.raises(ValueError, match="reward_dim and q_dim"):
        load_flux2_fact_trained_checkpoint(
            checkpoint,
            device="cpu",
            dtype=torch.float32,
        )


def test_dino_checkpoint_architecture_is_recorded_not_shape_inferred(tmp_path):
    project = tmp_path / "experiment"
    transformer = project / "models" / "step_1" / "transformer"
    transformer.mkdir(parents=True)
    checkpoint = transformer / "diffusion_pytorch_model.bin"
    expected = Flux2FACTModel(
        _tiny_params(),
        action_dim=6,
        state_dim=5,
        max_horizon=8,
        dino_dim=12,
    )
    torch.save(expected.state_dict(), checkpoint)
    (project / "model_config.json").write_text(
        json.dumps(
            {
                "models": {
                    "params": asdict(_tiny_params()),
                    "action_dim": 6,
                    "state_dim": 5,
                    "reward_dim": 1,
                    "success_dim": 1,
                    "q_dim": 1,
                    "reward_head_type": "direct",
                    "max_horizon": 8,
                    "dino_dim": 12,
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
    assert report.model_config.dino_dim == 12
    assert actual.dino_dim == 12
