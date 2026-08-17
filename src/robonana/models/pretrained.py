"""Load the official FLUX.2 checkpoint into the thin RoboNana subclass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from flux2.model import Flux2Params, Klein4BParams

from .flux2_fact import Flux2FACTModel


ROBOT_MODULE_NAMES = (
    "action_in",
    "state_in",
    "value_in",
    "horizon_embed",
    "segment_embed",
    "action_out",
    "state_out",
    "value_out",
)


@dataclass(frozen=True)
class PretrainedLoadReport:
    checkpoint: str
    checkpoint_parameters: int
    initialized_robot_parameters: tuple[str, ...]


def robot_parameter_names(model: Flux2FACTModel) -> tuple[str, ...]:
    prefixes = tuple(f"{name}." for name in ROBOT_MODULE_NAMES)
    return tuple(name for name, _ in model.named_parameters() if name.startswith(prefixes))


def configure_trainable_parameters(model: Flux2FACTModel, mode: str) -> tuple[str, ...]:
    """Select either a full shared-DiT update or the low-memory wiring smoke mode."""

    if mode not in {"full", "adapters"}:
        raise ValueError(f"train mode must be 'full' or 'adapters', got {mode!r}")
    robot_names = set(robot_parameter_names(model))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(mode == "full" or name in robot_names)
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def load_flux2_fact_checkpoint(
    checkpoint_path: str | Path,
    *,
    action_dim: int,
    state_dim: int,
    value_dim: int = 1,
    max_horizon: int = 64,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    params: Flux2Params | None = None,
) -> tuple[Flux2FACTModel, PretrainedLoadReport]:
    """Use FLUX.2's own meta-device loading pattern and initialize only new modules."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"FLUX.2 checkpoint not found: {path}")
    device = torch.device(device)
    params = Klein4BParams() if params is None else params

    with torch.device("meta"):
        model = Flux2FACTModel(
            params,
            action_dim=action_dim,
            state_dim=state_dim,
            value_dim=value_dim,
            max_horizon=max_horizon,
        ).to(dtype=dtype)

    state_dict = load_file(str(path), device=str(device))
    checkpoint_parameters = sum(tensor.numel() for tensor in state_dict.values())
    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    expected_missing = set(robot_parameter_names(model))
    actual_missing = set(incompatible.missing_keys)
    if incompatible.unexpected_keys or actual_missing != expected_missing:
        raise RuntimeError(
            "checkpoint is not an exact official FLUX.2 backbone: "
            f"missing={sorted(actual_missing)}, unexpected={sorted(incompatible.unexpected_keys)}"
        )

    for module_name in ROBOT_MODULE_NAMES:
        module = getattr(model, module_name)
        module.to_empty(device=device)
        module.reset_parameters()

    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(f"parameters remained on the meta device: {meta_parameters}")

    report = PretrainedLoadReport(
        checkpoint=str(path),
        checkpoint_parameters=checkpoint_parameters,
        initialized_robot_parameters=tuple(sorted(expected_missing)),
    )
    return model, report
