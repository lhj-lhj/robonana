"""Thin compatibility wrapper around FACT's RoboTwin client adapter."""

from __future__ import annotations


def _patch_warp_torch_namespace() -> None:
    """Restore the namespace expected by CuRobo 0.7.8 under Warp 1.16+."""

    try:
        import warp
    except ImportError:
        return
    if hasattr(warp, "torch"):
        return
    try:
        from warp._src import torch as warp_torch
    except ImportError:
        return
    warp.torch = warp_torch


_patch_warp_torch_namespace()

from evaluation.robotwin.model2robotwin_interface import (  # noqa: E402,F401
    eval,
    get_model,
    reset_model,
)

__all__ = ["eval", "get_model", "reset_model"]
