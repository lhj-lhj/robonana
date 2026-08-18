"""Small W&B boundary shared by smoke tests and the future FACT trainer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


WANDB_MODES = ("online", "offline", "disabled")


def init_wandb(
    *,
    project: str,
    entity: str,
    run_name: str | None,
    mode: str,
    config: Mapping[str, Any],
):
    """Initialize W&B without ever accepting or persisting an API key.

    Authentication belongs to ``wandb login`` or ``WANDB_API_KEY`` in the
    runtime environment. Keeping credentials out of this API prevents them
    from being copied into configs, checkpoints, logs, Notion, or Git.
    """

    if mode not in WANDB_MODES:
        raise ValueError(f"wandb mode must be one of {WANDB_MODES}, got {mode!r}")
    if mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B tracking requested; install robonana[tracking]") from error

    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        mode=mode,
        config=dict(config),
    )


def log_wandb(run, metrics: Mapping[str, Any], *, step: int) -> None:
    if run is not None:
        run.log(dict(metrics), step=step)


def finish_wandb(run, *, exit_code: int = 0) -> None:
    if run is not None:
        run.finish(exit_code=exit_code)
