"""Schema for the one maintained fixed-48 MAC checkpoint format."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from flux2.model import Flux2Params


@dataclass(frozen=True)
class RoboNanaCheckpointConfig:
    params: Flux2Params
    action_dim: int
    state_dim: int
    reward_dim: int
    success_dim: int
    q_dim: int
    reward_head_type: str
    max_horizon: int
    dino_dim: int | None
    pred_action_bidirectional: bool
    architecture_version: str
    chunk_horizon: int
    value_dim: int
    source: str
    expert_hidden_dim: int = 1024


def discover_model_config(checkpoint_path: str | Path) -> Path | None:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    for directory in [checkpoint.parent, *checkpoint.parents[1:7]]:
        for filename in ("model_config.json", "config.json"):
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    return None


def _model_section(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    models = payload.get("models", payload)
    if not isinstance(models, Mapping):
        raise ValueError("model config must contain a mapping under 'models'")
    nested = models.get("train")
    return nested if isinstance(nested, Mapping) else models


def _load_complete_config(path: Path) -> RoboNanaCheckpointConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = _model_section(payload)
    raw_params = models.get("params")
    if not isinstance(raw_params, Mapping):
        raise ValueError(f"model config is missing complete models.params: {path}")
    missing = sorted({field.name for field in fields(Flux2Params)} - set(raw_params))
    if missing:
        raise ValueError(f"model config is missing Flux2 params: {', '.join(missing)}")
    required = ("action_dim", "state_dim", "reward_dim", "success_dim", "q_dim", "max_horizon")
    missing = [name for name in required if name not in models]
    if missing:
        raise ValueError(f"model config is missing model dimensions: {', '.join(missing)}")
    architecture = str(models.get("architecture_version", ""))
    if architecture != "mac_mot_v2":
        raise ValueError("only architecture_version='mac_mot_v2' is supported")
    horizon = int(models.get("chunk_horizon", 0))
    if horizon != 48 or int(models["max_horizon"]) != 48:
        raise ValueError("the maintained model requires max_horizon=chunk_horizon=48")
    if str(models.get("reward_head_type")) != "binary_chunk" or int(models["reward_dim"]) != 48:
        raise ValueError("the maintained model requires a 48-logit binary_chunk reward head")
    if any(int(models[name]) != 1 for name in ("success_dim", "q_dim", "value_dim")):
        raise ValueError("success, Q and Value dimensions must be one")
    return RoboNanaCheckpointConfig(
        params=Flux2Params(**dict(raw_params)), action_dim=int(models["action_dim"]),
        state_dim=int(models["state_dim"]), reward_dim=48,
        success_dim=1, q_dim=1, reward_head_type="binary_chunk", max_horizon=48,
        dino_dim=None if models.get("dino_dim") is None else int(models["dino_dim"]),
        pred_action_bidirectional=True, architecture_version=architecture,
        chunk_horizon=48, value_dim=1, source=str(path),
        expert_hidden_dim=int(models.get("expert_hidden_dim", 1024)),
    )


def resolve_checkpoint_config(
    checkpoint_path: str | Path, *, config_path: str | Path | None = None,
    params: Flux2Params | None = None, action_dim: int | None = None,
    state_dim: int | None = None, reward_dim: int | None = None,
    success_dim: int | None = None, q_dim: int | None = None,
    reward_head_type: str | None = None, max_horizon: int | None = None,
    dino_dim: int | None = None, pred_action_bidirectional: bool | None = None,
    architecture_version: str | None = None, chunk_horizon: int | None = None,
    value_dim: int | None = None, expert_hidden_dim: int | None = None,
) -> RoboNanaCheckpointConfig:
    """Read the complete MAC schema; explicit overrides are for test tooling only."""

    discovered = Path(config_path).expanduser().resolve() if config_path else discover_model_config(checkpoint_path)
    if discovered is None:
        explicit = (params, action_dim, state_dim, reward_dim, success_dim, q_dim, reward_head_type, max_horizon)
        if not all(value is not None for value in explicit):
            raise FileNotFoundError("complete mac_mot_v2 config is required beside the checkpoint")
        if architecture_version not in (None, "mac_mot_v2"):
            raise ValueError("only architecture_version='mac_mot_v2' is supported")
        if int(max_horizon) != 48 or int(reward_dim) != 48 or int(success_dim) != 1 or int(q_dim) != 1:
            raise ValueError("explicit metadata must describe fixed-48 mac_mot_v2")
        return RoboNanaCheckpointConfig(
            params=params, action_dim=int(action_dim), state_dim=int(state_dim),
            reward_dim=int(reward_dim), success_dim=int(success_dim), q_dim=int(q_dim),
            reward_head_type=str(reward_head_type), max_horizon=int(max_horizon),
            dino_dim=dino_dim, pred_action_bidirectional=True,
            architecture_version="mac_mot_v2", chunk_horizon=48,
            value_dim=1 if value_dim is None else int(value_dim), source="explicit metadata",
            expert_hidden_dim=1024 if expert_hidden_dim is None else int(expert_hidden_dim),
        )
    config = _load_complete_config(discovered)
    for name, value in (("params", params), ("action_dim", action_dim), ("state_dim", state_dim),
                        ("reward_dim", reward_dim), ("success_dim", success_dim), ("q_dim", q_dim),
                        ("reward_head_type", reward_head_type), ("max_horizon", max_horizon),
                        ("dino_dim", dino_dim), ("chunk_horizon", chunk_horizon),
                        ("value_dim", value_dim), ("expert_hidden_dim", expert_hidden_dim)):
        if value is not None and value != getattr(config, name):
            raise ValueError(f"override {name} disagrees with the recorded mac_mot_v2 schema")
    return config
