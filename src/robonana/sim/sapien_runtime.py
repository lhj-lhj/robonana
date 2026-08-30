"""Explicit SAPIEN/Vulkan device selection for parallel RoboTwin clients."""

from __future__ import annotations

import os
from typing import Any


def configure_sapien_runtime(
    *,
    device: str | None = None,
    sapien_module: Any | None = None,
) -> tuple[str, str]:
    """Bind RoboTwin SAPIEN scenes to one CUDA-visible GPU and OIDN.

    SAPIEN 3's deprecated ``Engine`` wrapper ignores ``SapienRenderer`` and
    constructs ``RenderSystem()`` with the first physical Vulkan device.  That
    behavior can ignore ``CUDA_VISIBLE_DEVICES`` when selecting its default.
    Patch the wrapper before RoboTwin's startup render check creates a scene;
    callers should normally pass logical ``cuda:0`` after isolating one physical
    GPU with ``CUDA_VISIBLE_DEVICES``.
    """

    selected_device = str(
        device or os.environ.get("ROBONANA_SAPIEN_RENDER_DEVICE", "")
    ).strip()
    if not selected_device:
        raise ValueError("ROBONANA_SAPIEN_RENDER_DEVICE must be an explicit cuda:N alias")
    if not selected_device.startswith("cuda:") or not selected_device[5:].isdigit():
        raise ValueError(f"invalid SAPIEN render device {selected_device!r}; expected cuda:N")

    if sapien_module is None:
        import sapien as sapien_module

    scene_class = sapien_module.wrapper.scene.Scene
    configured_device = getattr(scene_class, "_robonana_render_device", None)
    if configured_device is not None:
        if configured_device != selected_device:
            raise RuntimeError(
                f"SAPIEN was already configured for {configured_device}, not {selected_device}"
            )
        return selected_device, "oidn"

    original_scene_init = scene_class.__init__
    pysapien = sapien_module.pysapien

    def scene_init(self, systems=None):
        if systems is None:
            systems = [
                pysapien.physx.PhysxCpuSystem(),
                pysapien.render.RenderSystem(selected_device),
            ]
        original_scene_init(self, systems)

    scene_class.__init__ = scene_init
    scene_class._robonana_render_device = selected_device

    original_set_denoiser = sapien_module.render.set_ray_tracing_denoiser

    def set_selected_denoiser(_requested: str) -> None:
        original_set_denoiser("oidn")

    sapien_module.render.set_ray_tracing_denoiser = set_selected_denoiser
    print(
        f"[RoboNana SAPIEN] render_device={selected_device} denoiser=oidn",
        flush=True,
    )
    return selected_device, "oidn"
