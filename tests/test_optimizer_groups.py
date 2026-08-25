import torch
from flux2.model import Flux2Params

from robonana.models.flux2_fact import Flux2FACTModel
from robonana.models.pretrained import robot_parameter_names
from robonana.training.robotwin_trainer import build_optimizer_param_groups


def _tiny_params() -> Flux2Params:
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


def test_pretrained_backbone_and_new_modules_use_distinct_learning_rates():
    model = Flux2FACTModel(
        _tiny_params(),
        action_dim=6,
        state_dim=5,
        value_dim=1,
        max_horizon=8,
        dino_dim=12,
    )
    groups = build_optimizer_param_groups(model, base_lr=2e-5, robot_lr=1e-4)
    groups_by_name = {group["name"]: group for group in groups}

    assert groups_by_name["flux_backbone"]["lr"] == 2e-5
    assert groups_by_name["robot_modules"]["lr"] == 1e-4

    parameter_name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    grouped_names = {
        group_name: {
            parameter_name_by_id[id(parameter)] for parameter in group["params"]
        }
        for group_name, group in groups_by_name.items()
    }
    expected_robot_names = set(robot_parameter_names(model))
    assert grouped_names["robot_modules"] == expected_robot_names
    assert grouped_names["flux_backbone"].isdisjoint(expected_robot_names)
    assert grouped_names["flux_backbone"] | grouped_names["robot_modules"] == set(
        parameter_name_by_id.values()
    )


def test_optimizer_grouping_respects_frozen_backbone():
    model = Flux2FACTModel(
        _tiny_params(),
        action_dim=6,
        state_dim=5,
        max_horizon=8,
    )
    robot_names = set(robot_parameter_names(model))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in robot_names)

    groups = build_optimizer_param_groups(model, base_lr=2e-5, robot_lr=1e-4)

    assert [group["name"] for group in groups] == ["robot_modules"]
    assert groups[0]["lr"] == 1e-4
    assert all(isinstance(parameter, torch.nn.Parameter) for parameter in groups[0]["params"])
