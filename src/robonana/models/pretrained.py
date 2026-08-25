"""Load the official FLUX.2 checkpoint into the thin RoboNana subclass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from flux2.model import Flux2Params, Klein4BParams

from .checkpoint_config import RoboNanaCheckpointConfig, resolve_checkpoint_config
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
OPTIONAL_DINO_MODULE_NAMES = (
    "dino_in",
    "dino_out",
    "dino_segment_embed",
)


def robot_module_names(model: Flux2FACTModel) -> tuple[str, ...]:
    return ROBOT_MODULE_NAMES + tuple(
        name for name in OPTIONAL_DINO_MODULE_NAMES if hasattr(model, name)
    )


@dataclass(frozen=True)
class PretrainedLoadReport:
    checkpoint: str
    checkpoint_parameters: int
    initialized_robot_parameters: tuple[str, ...]
    model_config: RoboNanaCheckpointConfig | None = None


def robot_parameter_names(model: Flux2FACTModel) -> tuple[str, ...]:
    prefixes = tuple(f"{name}." for name in robot_module_names(model))
    return tuple(name for name, _ in model.named_parameters() if name.startswith(prefixes))


def configure_trainable_parameters(model: Flux2FACTModel, mode: str) -> tuple[str, ...]:
    """Select either a full shared-DiT update or the low-memory wiring smoke mode."""

    if mode not in {"full", "adapters"}:
        raise ValueError(f"train mode must be 'full' or 'adapters', got {mode!r}")
    robot_names = set(robot_parameter_names(model))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(mode == "full" or name in robot_names)
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def initialize_flux2_fact_model(
    *,
    action_dim: int,
    state_dim: int,
    value_dim: int = 1,
    max_horizon: int = 64,
    dino_dim: int | None = None,
    pred_action_bidirectional: bool = False,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    params: Flux2Params,
) -> Flux2FACTModel:
    """Initialize a scratch RoboNana model from the official FLUX.2 modules."""

    model = Flux2FACTModel(
        params,
        action_dim=action_dim,
        state_dim=state_dim,
        value_dim=value_dim,
        max_horizon=max_horizon,
        dino_dim=dino_dim,
        pred_action_bidirectional=pred_action_bidirectional,
    )
    return model.to(device=torch.device(device), dtype=dtype)


def load_flux2_fact_checkpoint(
    checkpoint_path: str | Path,
    *,
    action_dim: int,
    state_dim: int,
    value_dim: int = 1,
    max_horizon: int = 64,
    dino_dim: int | None = None,
    pred_action_bidirectional: bool = False,
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
            dino_dim=dino_dim,
            pred_action_bidirectional=pred_action_bidirectional,
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

    for module_name in robot_module_names(model):
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


def load_flux2_fact_trained_checkpoint(
    checkpoint_path: str | Path,
    *,
    action_dim: int | None = None,
    state_dim: int | None = None,
    value_dim: int | None = None,
    max_horizon: int | None = None,
    dino_dim: int | None = None,
    pred_action_bidirectional: bool | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    params: Flux2Params | None = None,
    config_path: str | Path | None = None,
) -> tuple[Flux2FACTModel, PretrainedLoadReport]:
    """Load a full FACT-exported RoboNana transformer without duplicating weights."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"RoboNana checkpoint not found: {path}")
    device = torch.device(device)
    model_config = resolve_checkpoint_config(
        path,
        config_path=config_path,
        params=params,
        action_dim=action_dim,
        state_dim=state_dim,
        value_dim=value_dim,
        max_horizon=max_horizon,
        dino_dim=dino_dim,
        pred_action_bidirectional=pred_action_bidirectional,
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True, mmap=True)

    with torch.device("meta"):
        model = Flux2FACTModel(
            model_config.params,
            action_dim=model_config.action_dim,
            state_dim=model_config.state_dim,
            value_dim=model_config.value_dim,
            max_horizon=model_config.max_horizon,
            dino_dim=model_config.dino_dim,
            pred_action_bidirectional=model_config.pred_action_bidirectional,
        ).to(dtype=dtype)

    checkpoint_parameters = sum(tensor.numel() for tensor in state_dict.values())
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "trained checkpoint does not exactly match RoboNana: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.to(device=device, dtype=dtype)
    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(f"parameters remained on the meta device: {meta_parameters}")
    return model, PretrainedLoadReport(
        checkpoint=str(path),
        checkpoint_parameters=checkpoint_parameters,
        initialized_robot_parameters=(),
        model_config=model_config,
    )
