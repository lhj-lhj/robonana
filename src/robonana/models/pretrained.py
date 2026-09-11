"""Load the single maintained mac_mot_v2 FLUX.2 model.

The old 120k legacy model was converted once to a complete MAC checkpoint.
Runtime code intentionally has no legacy architecture or variable-horizon
checkpoint loader; pass the converted 1000-step checkpoint (or a later MAC
checkpoint) through ``load_flux2_fact_trained_checkpoint``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from flux2.model import Flux2Params

from .checkpoint_config import RoboNanaCheckpointConfig, resolve_checkpoint_config
from .mac_flux2_fact import MacFlux2FACTModel


MAC_MODULE_NAMES = (
    "action_in", "state_in", "reward_token", "success_token", "action_out",
    "state_out", "reward_out", "success_out", "actor_world_segment_embed",
    "value_expert", "q_expert",
)


@dataclass(frozen=True)
class PretrainedLoadReport:
    checkpoint: str
    checkpoint_parameters: int
    initialized_robot_parameters: tuple[str, ...] = ()
    model_config: RoboNanaCheckpointConfig | None = None
    loaded_parameter_names: tuple[str, ...] = ()
    skipped_checkpoint_parameters: tuple[str, ...] = ()


def robot_module_names(model: MacFlux2FACTModel) -> tuple[str, ...]:
    return tuple(name for name in MAC_MODULE_NAMES if hasattr(model, name))


def robot_parameter_names(model: MacFlux2FACTModel) -> tuple[str, ...]:
    prefixes = tuple(f"{name}." for name in robot_module_names(model))
    return tuple(name for name, _ in model.named_parameters() if name.startswith(prefixes))


def configure_trainable_parameters(model: MacFlux2FACTModel, mode: str) -> tuple[str, ...]:
    """Select the only supported optimizer surfaces for the fixed MAC model."""

    if not isinstance(model, MacFlux2FACTModel):
        raise TypeError("RoboNana only supports MacFlux2FACTModel")
    if mode not in {"world_policy", "critic"}:
        raise ValueError("mac_mot_v2 train_mode must be world_policy or critic")
    return model.set_training_phase(mode)


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"RoboNana checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint must contain a state-dict mapping: {path}")
    return state


def load_flux2_backbone_checkpoint(checkpoint_path, *, params, action_dim=14,
                                   state_dim=14, expert_hidden_dim=1024,
                                   device="cpu", dtype=torch.bfloat16):
    """中文：原始 FLUX 初始化，不接受缺失 backbone 或夹带旧 robot head。

    English: Load only the complete upstream FLUX state, then reuse the existing
    ImageWAM-derived expert transfer (queries/scalar heads stay freshly initialized).
    Upstream loader: https://github.com/black-forest-labs/flux2/blob/main/src/flux2/util.py
    No transformer implementation is copied or precision policy changed here.
    """
    from safetensors.torch import load_file
    from .flux2_scalar_expert import initialize_scalar_expert_from_flux

    model = MacFlux2FACTModel(params, action_dim=action_dim, state_dim=state_dim,
                             expert_hidden_dim=expert_hidden_dim)
    state = load_file(str(checkpoint_path), device="cpu")
    robot_names = set(robot_parameter_names(model))
    backbone = set(model.state_dict()) - robot_names
    if set(state) != backbone:
        raise ValueError(f"Original FLUX keys mismatch: missing={sorted(backbone-set(state))}, "
                         f"unexpected={sorted(set(state)-backbone)}")
    model.load_state_dict(state, strict=False)  # Exact key set checked above; shapes checked by PyTorch.
    for name in ("value_expert", "q_expert"):
        initialize_scalar_expert_from_flux(getattr(model, name), model)
    model.to(device=device, dtype=dtype)
    return model, PretrainedLoadReport(
        checkpoint=str(checkpoint_path), checkpoint_parameters=sum(v.numel() for v in state.values()),
        initialized_robot_parameters=tuple(sorted(robot_names)), loaded_parameter_names=tuple(sorted(state)))


def load_flux2_fact_trained_checkpoint(
    checkpoint_path: str | Path,
    *,
    action_dim: int | None = None,
    state_dim: int | None = None,
    reward_dim: int | None = None,
    success_dim: int | None = None,
    q_dim: int | None = None,
    reward_head_type: str | None = None,
    max_horizon: int | None = None,
    dino_dim: int | None = None,
    pred_action_bidirectional: bool | None = None,
    architecture_version: str | None = None,
    chunk_horizon: int | None = None,
    value_dim: int | None = None,
    expert_hidden_dim: int | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    params: Flux2Params | None = None,
    config_path: str | Path | None = None,
) -> tuple[MacFlux2FACTModel, PretrainedLoadReport]:
    """Strictly load a complete fixed-48 MAC checkpoint and its recorded schema."""

    path = Path(checkpoint_path).expanduser().resolve()
    state_dict = _load_state(path)
    config = resolve_checkpoint_config(
        path, config_path=config_path, params=params, action_dim=action_dim,
        state_dim=state_dim, reward_dim=reward_dim, success_dim=success_dim,
        q_dim=q_dim, reward_head_type=reward_head_type, max_horizon=max_horizon,
        dino_dim=dino_dim, pred_action_bidirectional=pred_action_bidirectional,
        architecture_version=architecture_version, chunk_horizon=chunk_horizon,
        value_dim=value_dim, expert_hidden_dim=expert_hidden_dim,
    )
    if config.architecture_version != "mac_mot_v2":
        raise ValueError("the maintained checkpoint format is mac_mot_v2 only")
    with torch.device("meta"):
        model = MacFlux2FACTModel(
            config.params,
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            chunk_horizon=config.chunk_horizon,
            reward_dim=config.reward_dim,
            success_dim=config.success_dim,
            q_dim=config.q_dim,
            value_dim=config.value_dim,
            dino_dim=config.dino_dim,
            expert_hidden_dim=config.expert_hidden_dim,
        ).to(dtype=dtype)
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "trained MAC checkpoint does not exactly match its recorded schema: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.to(device=torch.device(device), dtype=dtype)
    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(f"parameters remained on the meta device: {meta_parameters}")
    return model, PretrainedLoadReport(
        checkpoint=str(path),
        checkpoint_parameters=sum(tensor.numel() for tensor in state_dict.values()),
        model_config=config,
        loaded_parameter_names=tuple(sorted(state_dict)),
    )


__all__ = [
    "PretrainedLoadReport", "configure_trainable_parameters",
    "load_flux2_fact_trained_checkpoint", "robot_module_names", "robot_parameter_names",
]
