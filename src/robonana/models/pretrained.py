"""Load the official FLUX.2 checkpoint into the thin RoboNana subclass."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import warnings

import torch
from safetensors.torch import load_file

from flux2.model import Flux2Params, Klein4BParams

from .checkpoint_config import (
    RoboNanaCheckpointConfig,
    discover_model_config,
    resolve_checkpoint_config,
)
from .flux2_fact import Flux2FACTModel, MacFlux2FACTModel


ROBOT_MODULE_NAMES = (
    "action_in",
    "state_in",
    "reward_token",
    "success_token",
    "q_in",
    "horizon_embed",
    "segment_embed",
    "q_segment_embed",
    "action_out",
    "state_out",
    "reward_out",
    "success_out",
    "q_out",
)
REWARD_Q_MODULE_NAMES = (
    "reward_token",
    "success_token",
    "q_in",
    "q_segment_embed",
    "reward_out",
    "success_out",
    "q_out",
)
DIRECT_REWARD_SUCCESS_MODULE_NAMES = (
    "reward_token",
    "success_token",
    "reward_out",
    "success_out",
)
OPTIONAL_DINO_MODULE_NAMES = (
    "dino_in",
    "dino_out",
    "dino_segment_embed",
)
MAC_MODULE_NAMES = (
    "value_token",
    "q_token",
    "mac_segment_embed",
    "value_out",
)


def robot_module_names(model: Flux2FACTModel) -> tuple[str, ...]:
    candidates = ROBOT_MODULE_NAMES + MAC_MODULE_NAMES + OPTIONAL_DINO_MODULE_NAMES
    return tuple(
        name for name in dict.fromkeys(candidates) if hasattr(model, name)
    )


@dataclass(frozen=True)
class PretrainedLoadReport:
    checkpoint: str
    checkpoint_parameters: int
    initialized_robot_parameters: tuple[str, ...]
    model_config: RoboNanaCheckpointConfig | None = None
    loaded_parameter_names: tuple[str, ...] = ()
    skipped_checkpoint_parameters: tuple[str, ...] = ()


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
    reward_dim: int = 1,
    success_dim: int = 1,
    q_dim: int = 1,
    max_horizon: int = 64,
    dino_dim: int | None = None,
    pred_action_bidirectional: bool = False,
    architecture_version: str = "legacy_v1",
    chunk_horizon: int = 48,
    value_dim: int = 1,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    params: Flux2Params,
) -> Flux2FACTModel:
    """Initialize a scratch RoboNana model from the official FLUX.2 modules."""

    if architecture_version == "mac_v1":
        model = MacFlux2FACTModel(
            params,
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_horizon=chunk_horizon,
            reward_dim=reward_dim,
            success_dim=success_dim,
            q_dim=q_dim,
            value_dim=value_dim,
            dino_dim=dino_dim,
        )
    elif architecture_version == "legacy_v1":
        model = Flux2FACTModel(
            params,
            action_dim=action_dim,
            state_dim=state_dim,
            reward_dim=reward_dim,
            success_dim=success_dim,
            q_dim=q_dim,
            max_horizon=max_horizon,
            dino_dim=dino_dim,
            pred_action_bidirectional=pred_action_bidirectional,
        )
    else:
        raise ValueError(f"unsupported architecture_version: {architecture_version}")
    return model.to(device=torch.device(device), dtype=dtype)


def load_flux2_fact_checkpoint(
    checkpoint_path: str | Path,
    *,
    action_dim: int,
    state_dim: int,
    reward_dim: int = 1,
    success_dim: int = 1,
    q_dim: int = 1,
    max_horizon: int = 64,
    dino_dim: int | None = None,
    pred_action_bidirectional: bool = False,
    architecture_version: str = "legacy_v1",
    chunk_horizon: int = 48,
    value_dim: int = 1,
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
        if architecture_version == "mac_v1":
            model = MacFlux2FACTModel(
                params,
                action_dim=action_dim,
                state_dim=state_dim,
                reward_dim=reward_dim,
                success_dim=success_dim,
                q_dim=q_dim,
                value_dim=value_dim,
                chunk_horizon=chunk_horizon,
                dino_dim=dino_dim,
            ).to(dtype=dtype)
        elif architecture_version == "legacy_v1":
            model = Flux2FACTModel(
                params,
                action_dim=action_dim,
                state_dim=state_dim,
                reward_dim=reward_dim,
                success_dim=success_dim,
                q_dim=q_dim,
                max_horizon=max_horizon,
                dino_dim=dino_dim,
                pred_action_bidirectional=pred_action_bidirectional,
            ).to(dtype=dtype)
        else:
            raise ValueError(f"unsupported architecture_version: {architecture_version}")

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
    has_recorded_config = config_path is not None or discover_model_config(path) is not None
    source_config = resolve_checkpoint_config(
        path,
        config_path=config_path,
        params=params,
        action_dim=action_dim,
        state_dim=state_dim,
        reward_dim=reward_dim,
        success_dim=None if has_recorded_config else success_dim,
        q_dim=q_dim,
        reward_head_type=None if has_recorded_config else reward_head_type,
        max_horizon=max_horizon,
        dino_dim=dino_dim,
        pred_action_bidirectional=pred_action_bidirectional,
        architecture_version=architecture_version,
        chunk_horizon=chunk_horizon,
        value_dim=value_dim,
    )
    model_config = replace(
        source_config,
        success_dim=(source_config.success_dim if success_dim is None else int(success_dim)),
        reward_head_type=(
            source_config.reward_head_type
            if reward_head_type is None
            else str(reward_head_type)
        ),
    )
    if model_config.architecture_version == "legacy_v1" and (
        model_config.reward_head_type != "direct" or model_config.success_dim != 1
    ):
        raise ValueError(
            "current RoboNana requires reward_head_type='direct' and success_dim=1; "
            "pass those explicit target settings to warm-start a legacy flow-reward checkpoint"
        )
    state_dict = torch.load(path, map_location="cpu", weights_only=True, mmap=True)

    with torch.device("meta"):
        if model_config.architecture_version == "mac_v1":
            model = MacFlux2FACTModel(
                model_config.params,
                action_dim=model_config.action_dim,
                state_dim=model_config.state_dim,
                chunk_horizon=model_config.chunk_horizon,
                reward_dim=model_config.reward_dim,
                success_dim=model_config.success_dim,
                q_dim=model_config.q_dim,
                value_dim=model_config.value_dim,
                dino_dim=model_config.dino_dim,
            ).to(dtype=dtype)
        else:
            model = Flux2FACTModel(
                model_config.params,
                action_dim=model_config.action_dim,
                state_dim=model_config.state_dim,
                reward_dim=model_config.reward_dim,
                success_dim=model_config.success_dim,
                q_dim=model_config.q_dim,
                max_horizon=model_config.max_horizon,
                dino_dim=model_config.dino_dim,
                pred_action_bidirectional=model_config.pred_action_bidirectional,
            ).to(dtype=dtype)

    checkpoint_parameters = sum(tensor.numel() for tensor in state_dict.values())
    legacy_value_keys = tuple(
        name
        for name in state_dict
        if name.startswith("value_in.") or name.startswith("value_out.")
    )
    initialized_robot_parameters: tuple[str, ...] = ()
    if model_config.architecture_version == "mac_v1":
        incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "trained MAC checkpoint does not exactly match its recorded schema: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
    elif legacy_value_keys:
        warnings.warn(
            "Legacy RoboNana value_in/value_out weights are intentionally skipped; "
            "direct reward/success and q_in/q_out are newly initialized and no "
            "time-to-go weights are mapped to Q.",
            UserWarning,
            stacklevel=2,
        )
        filtered_state_dict = {
            name: tensor for name, tensor in state_dict.items() if name not in legacy_value_keys
        }
        incompatible = model.load_state_dict(filtered_state_dict, strict=False, assign=True)
        expected_missing = {
            name
            for name, _ in model.named_parameters()
            if name.split(".", 1)[0] in REWARD_Q_MODULE_NAMES
        }
        actual_missing = set(incompatible.missing_keys)
        if incompatible.unexpected_keys or actual_missing != expected_missing:
            raise RuntimeError(
                "legacy trained checkpoint does not match RoboNana outside value heads: "
                f"missing={sorted(actual_missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        for module_name in REWARD_Q_MODULE_NAMES:
            module = getattr(model, module_name)
            module.to_empty(device=device)
            module.reset_parameters()
        initialized_robot_parameters = tuple(sorted(expected_missing))
    elif source_config.reward_head_type == "flow":
        legacy_reward_keys = {
            name
            for name in state_dict
            if name.startswith("reward_in.") or name.startswith("reward_out.")
        }
        warnings.warn(
            "Legacy flow-matched reward_in/reward_out weights are intentionally skipped; "
            "the direct reward query/head and success classifier are newly initialized. "
            "Backbone, action, state, Q, image, and DINO weights are preserved.",
            UserWarning,
            stacklevel=2,
        )
        filtered_state_dict = {
            name: tensor for name, tensor in state_dict.items() if name not in legacy_reward_keys
        }
        incompatible = model.load_state_dict(filtered_state_dict, strict=False, assign=True)
        expected_missing = {
            name
            for name, _ in model.named_parameters()
            if name.split(".", 1)[0] in DIRECT_REWARD_SUCCESS_MODULE_NAMES
        }
        actual_missing = set(incompatible.missing_keys)
        if incompatible.unexpected_keys or actual_missing != expected_missing:
            raise RuntimeError(
                "flow-reward checkpoint does not match RoboNana outside reward/success: "
                f"missing={sorted(actual_missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        for module_name in DIRECT_REWARD_SUCCESS_MODULE_NAMES:
            module = getattr(model, module_name)
            module.to_empty(device=device)
            module.reset_parameters()
        initialized_robot_parameters = tuple(sorted(expected_missing))
    else:
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
        initialized_robot_parameters=initialized_robot_parameters,
        model_config=model_config,
    )


LEGACY_PROJECT_MODULE_NAMES = frozenset(
    {
        *ROBOT_MODULE_NAMES,
        *OPTIONAL_DINO_MODULE_NAMES,
        "value_in",
        "value_out",
        "reward_in",
        "dino_segment_embed",
    }
)
MAC_COMPATIBLE_ROBOT_MODULE_NAMES = frozenset(
    {"action_in", "action_out", "state_in", "state_out", "dino_in", "dino_out"}
)


def load_mac_from_legacy_checkpoint(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    action_dim: int,
    state_dim: int,
    reward_dim: int = 48,
    success_dim: int = 1,
    q_dim: int = 1,
    value_dim: int = 1,
    chunk_horizon: int = 48,
    dino_dim: int | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    params: Flux2Params | None = None,
) -> tuple[MacFlux2FACTModel, PretrainedLoadReport]:
    """Warm-start MAC from 120k while excluding obsolete project heads.

    The whitelist preserves the official FLUX.2 backbone and the compatible
    robot action/state/DINO projections.  Horizon, segment, reward, success,
    Value, and flow-Q parameters are always reinitialized, even if a source key
    happens to have the same shape as a new MAC parameter.
    """

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"legacy RoboNana checkpoint not found: {path}")
    source = resolve_checkpoint_config(path, config_path=config_path)
    if source.architecture_version != "legacy_v1":
        raise ValueError("load_mac_from_legacy_checkpoint requires a legacy_v1 source")
    target_params = source.params if params is None else params
    if target_params != source.params:
        raise ValueError("MAC target FLUX.2 params must exactly match the 120k source")
    device = torch.device(device)
    state_dict = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    with torch.device("meta"):
        model = MacFlux2FACTModel(
            target_params,
            action_dim=action_dim,
            state_dim=state_dim,
            reward_dim=reward_dim,
            success_dim=success_dim,
            q_dim=q_dim,
            value_dim=value_dim,
            chunk_horizon=chunk_horizon,
            dino_dim=dino_dim,
        ).to(dtype=dtype)

    target_state = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for name, tensor in state_dict.items():
        root = name.split(".", 1)[0]
        allowed = (
            root not in LEGACY_PROJECT_MODULE_NAMES
            or root in MAC_COMPATIBLE_ROBOT_MODULE_NAMES
        )
        target_tensor = target_state.get(name)
        if (
            allowed
            and target_tensor is not None
            and tuple(target_tensor.shape) == tuple(tensor.shape)
        ):
            loaded[name] = tensor
        else:
            skipped.append(name)

    incompatible = model.load_state_dict(loaded, strict=False, assign=True)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected keys during 120k MAC migration: {incompatible.unexpected_keys}"
        )
    missing = tuple(incompatible.missing_keys)
    loaded_roots = {name.split(".", 1)[0] for name in loaded}
    missing_roots = {name.split(".", 1)[0] for name in missing}
    partially_loaded = loaded_roots & missing_roots
    if partially_loaded:
        raise RuntimeError(
            "cannot safely initialize partially loaded modules: "
            + ", ".join(sorted(partially_loaded))
        )
    for module_name in sorted(missing_roots):
        module = getattr(model, module_name, None)
        if module is None or not hasattr(module, "reset_parameters"):
            raise RuntimeError(
                f"missing non-initializable target module during migration: {module_name}"
            )
        module.to_empty(device=device)
        module.reset_parameters()

    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(f"parameters remained on the meta device: {meta_parameters}")
    model.to(device=device, dtype=dtype)
    checkpoint_parameters = sum(tensor.numel() for tensor in state_dict.values())
    initialized = tuple(sorted(name for name in missing if name in dict(model.named_parameters())))
    target_config = replace(
        source,
        action_dim=int(action_dim),
        state_dim=int(state_dim),
        reward_dim=int(reward_dim),
        success_dim=int(success_dim),
        q_dim=int(q_dim),
        reward_head_type="binary_chunk",
        max_horizon=int(chunk_horizon),
        dino_dim=None if dino_dim is None else int(dino_dim),
        pred_action_bidirectional=True,
        legacy_value_dim=None,
        architecture_version="mac_v1",
        chunk_horizon=int(chunk_horizon),
        value_dim=int(value_dim),
        source=f"MAC migration from {Path(config_path).expanduser().resolve()}",
    )
    warnings.warn(
        "Loaded the legacy 120k FLUX/action/state/image weights into mac_v1; "
        f"skipped {len(skipped)} obsolete project tensors and initialized "
        f"{len(initialized)} MAC tensors.",
        UserWarning,
        stacklevel=2,
    )
    return model, PretrainedLoadReport(
        checkpoint=str(path),
        checkpoint_parameters=checkpoint_parameters,
        initialized_robot_parameters=initialized,
        model_config=target_config,
        loaded_parameter_names=tuple(sorted(loaded)),
        skipped_checkpoint_parameters=tuple(sorted(skipped)),
    )
