"""Optimizer parameter groups for pretrained FLUX RoboNana training."""

from __future__ import annotations

from typing import Any

import torch

from robonana.models.pretrained import robot_parameter_names


def build_optimizer_param_groups(
    model: torch.nn.Module,
    *,
    base_lr: float,
    robot_lr: float,
) -> list[dict[str, Any]]:
    """Split trainable tensors into pretrained FLUX and new RoboNana modules."""

    if base_lr <= 0 or robot_lr <= 0:
        raise ValueError("base_lr and robot_lr must both be positive")
    robot_names = set(robot_parameter_names(model))
    buckets: dict[str, list[torch.nn.Parameter]] = {
        "flux_backbone": [],
        "robot_modules": [],
    }
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        group_name = "robot_modules" if name in robot_names else "flux_backbone"
        buckets[group_name].append(parameter)

    param_groups = [
        {"name": "flux_backbone", "params": buckets["flux_backbone"], "lr": float(base_lr)},
        {"name": "robot_modules", "params": buckets["robot_modules"], "lr": float(robot_lr)},
    ]
    return [group for group in param_groups if group["params"]]
