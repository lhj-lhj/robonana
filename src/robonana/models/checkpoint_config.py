"""Load the exact RoboNana architecture recorded by a training run."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

from flux2.model import Flux2Params


@dataclass(frozen=True)
class RoboNanaCheckpointConfig:
    """Everything required to reconstruct a trained ``Flux2FACTModel``."""

    params: Flux2Params
    action_dim: int
    state_dim: int
    reward_dim: int
    q_dim: int
    max_horizon: int
    dino_dim: int | None
    pred_action_bidirectional: bool
    legacy_value_dim: int | None
    source: str


def _model_section(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    models = payload.get("models", payload)
    if not isinstance(models, Mapping):
        raise ValueError("model config must contain a mapping under 'models'")
    nested = models.get("train")
    if isinstance(nested, Mapping):
        models = nested
    return models


def discover_model_config(checkpoint_path: str | Path) -> Path | None:
    """Find FACT's project ``config.json`` above a transformer checkpoint."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    directories = [checkpoint.parent, *checkpoint.parents[1:7]]
    for directory in directories:
        for filename in ("model_config.json", "config.json"):
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    return None


def _load_complete_config(path: Path) -> RoboNanaCheckpointConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"model config must be a JSON object: {path}")
    models = _model_section(payload)

    raw_params = models.get("params")
    if not isinstance(raw_params, Mapping):
        raise ValueError(
            f"model config is missing the complete 'models.params' mapping: {path}"
        )
    param_names = {field.name for field in fields(Flux2Params)}
    missing_params = sorted(param_names.difference(raw_params))
    if missing_params:
        raise ValueError(
            f"model config {path} is missing Flux2 params: {', '.join(missing_params)}"
        )

    dimension_names = ("action_dim", "state_dim", "max_horizon")
    missing_dimensions = [name for name in dimension_names if name not in models]
    if missing_dimensions:
        raise ValueError(
            f"model config {path} is missing model dimensions: {', '.join(missing_dimensions)}"
        )

    raw_pred_action_bidirectional = models.get("pred_action_bidirectional", False)
    if not isinstance(raw_pred_action_bidirectional, bool):
        raise ValueError(
            "models.pred_action_bidirectional must be a JSON boolean in "
            f"model config: {path}"
        )

    has_reward_dim = "reward_dim" in models
    has_q_dim = "q_dim" in models
    if has_reward_dim != has_q_dim:
        raise ValueError(
            f"model config {path} must define reward_dim and q_dim together"
        )
    legacy_value_dim = None
    if has_reward_dim:
        reward_dim = int(models["reward_dim"])
        q_dim = int(models["q_dim"])
    elif "value_dim" in models:
        # This is an explicit legacy schema marker, not tensor-shape inference.
        legacy_value_dim = int(models["value_dim"])
        reward_dim = q_dim = 1
    else:
        raise ValueError(
            f"model config {path} is missing reward_dim and q_dim "
            "(or the explicit legacy value_dim marker)"
        )
    if reward_dim != 1 or q_dim != 1:
        raise ValueError("reward_dim and q_dim must both be one scalar token")

    return RoboNanaCheckpointConfig(
        params=Flux2Params(**dict(raw_params)),
        action_dim=int(models["action_dim"]),
        state_dim=int(models["state_dim"]),
        reward_dim=reward_dim,
        q_dim=q_dim,
        max_horizon=int(models["max_horizon"]),
        # Legacy checkpoints explicitly reconstruct the pre-DINO architecture.
        # We never infer this dimension from checkpoint tensor shapes.
        dino_dim=None if models.get("dino_dim") is None else int(models["dino_dim"]),
        # Configs written before the hybrid layout used causal A and causal G.
        pred_action_bidirectional=raw_pred_action_bidirectional,
        legacy_value_dim=legacy_value_dim,
        source=str(path),
    )


def resolve_checkpoint_config(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
    params: Flux2Params | None = None,
    action_dim: int | None = None,
    state_dim: int | None = None,
    reward_dim: int | None = None,
    q_dim: int | None = None,
    max_horizon: int | None = None,
    dino_dim: int | None = None,
    pred_action_bidirectional: bool | None = None,
) -> RoboNanaCheckpointConfig:
    """Resolve an exact architecture; never infer structure from model tensors."""

    discovered = (
        Path(config_path).expanduser().resolve()
        if config_path
        else discover_model_config(checkpoint_path)
    )
    if discovered is None:
        explicit = (params, action_dim, state_dim, reward_dim, q_dim, max_horizon)
        if any(value is not None for value in explicit) and not all(
            value is not None for value in explicit
        ):
            raise ValueError(
                "partial explicit model metadata is not allowed; provide params, action_dim, "
                "state_dim, reward_dim, q_dim, and max_horizon together"
            )
        if all(value is not None for value in explicit):
            resolved = RoboNanaCheckpointConfig(
                params=params,
                action_dim=int(action_dim),
                state_dim=int(state_dim),
                reward_dim=int(reward_dim),
                q_dim=int(q_dim),
                max_horizon=int(max_horizon),
                dino_dim=None if dino_dim is None else int(dino_dim),
                pred_action_bidirectional=(
                    False
                    if pred_action_bidirectional is None
                    else pred_action_bidirectional
                ),
                legacy_value_dim=None,
                source="explicit model metadata",
            )
            if resolved.reward_dim != 1 or resolved.q_dim != 1:
                raise ValueError("reward_dim and q_dim must both be one scalar token")
            return resolved
        raise FileNotFoundError(
            "no complete RoboNana model config was found above checkpoint "
            f"{Path(checkpoint_path).expanduser().resolve()}; pass --model-config explicitly"
        )
    if not discovered.is_file():
        raise FileNotFoundError(f"RoboNana model config not found: {discovered}")

    resolved = _load_complete_config(discovered)
    resolved = replace(
        resolved,
        params=params if params is not None else resolved.params,
        action_dim=int(action_dim) if action_dim is not None else resolved.action_dim,
        state_dim=int(state_dim) if state_dim is not None else resolved.state_dim,
        reward_dim=int(reward_dim) if reward_dim is not None else resolved.reward_dim,
        q_dim=int(q_dim) if q_dim is not None else resolved.q_dim,
        max_horizon=int(max_horizon) if max_horizon is not None else resolved.max_horizon,
        dino_dim=int(dino_dim) if dino_dim is not None else resolved.dino_dim,
        pred_action_bidirectional=(
            pred_action_bidirectional
            if pred_action_bidirectional is not None
            else resolved.pred_action_bidirectional
        ),
    )
    if resolved.reward_dim != 1 or resolved.q_dim != 1:
        raise ValueError("reward_dim and q_dim must both be one scalar token")
    return resolved
