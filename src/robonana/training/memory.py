"""Conservative GPU preflight for tests on a shared accelerator."""

from __future__ import annotations

from dataclasses import dataclass


GIB = 1 << 30


@dataclass(frozen=True)
class MemoryPreflight:
    mode: str
    checkpoint_bytes: int
    required_free_bytes: int
    free_bytes: int

    @property
    def can_run(self) -> bool:
        return self.free_bytes >= self.required_free_bytes


def estimate_required_free_bytes(checkpoint_bytes: int, mode: str, headroom_bytes: int = 2 * GIB) -> int:
    """Estimate a safe lower bound before allocating anything on a shared GPU.

    Adapter smoke mode needs one BF16 checkpoint plus activation headroom. Full
    AdamW training needs weights, gradients, and two optimizer moments before
    accounting for activations.
    """

    if checkpoint_bytes <= 0:
        raise ValueError("checkpoint_bytes must be positive")
    if headroom_bytes < 0:
        raise ValueError("headroom_bytes must be non-negative")
    if mode == "adapters":
        multiplier = 1
    elif mode == "full":
        multiplier = 4
    else:
        raise ValueError(f"train mode must be 'full' or 'adapters', got {mode!r}")
    return multiplier * checkpoint_bytes + headroom_bytes


def memory_preflight(
    *, checkpoint_bytes: int, free_bytes: int, mode: str, headroom_bytes: int = 2 * GIB
) -> MemoryPreflight:
    return MemoryPreflight(
        mode=mode,
        checkpoint_bytes=checkpoint_bytes,
        required_free_bytes=estimate_required_free_bytes(checkpoint_bytes, mode, headroom_bytes),
        free_bytes=free_bytes,
    )
