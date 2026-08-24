"""Explicit SAPIEN/Vulkan device selection for parallel RoboTwin clients."""

from __future__ import annotations

import os
from typing import Any


def configure_sapien_runtime(
    *,
    device: str | None = None,
    denoiser: str | None = None,
    sapien_module: Any | None = None,
) -> tuple[str, str]:
    """Bind default SAPIEN scenes to one physical GPU before RoboTwin imports.

    SAPIEN 3's deprecated ``Engine`` wrapper ignores ``SapienRenderer`` and
    constructs ``RenderSystem()`` with the first physical Vulkan device.  That
    behavior also ignores ``CUDA_VISIBLE_DEVICES``.  Patch the wrapper's default
    scene constructor before RoboTwin's startup render check creates a scene.
    """

    selected_device = str(
        device or os.environ.get("ROBONANA_SAPIEN_RENDER_DEVICE", "")
    ).strip()
    if not selected_device:
        raise ValueError("ROBONANA_SAPIEN_RENDER_DEVICE must be an explicit cuda:N alias")
    if not selected_device.startswith("cuda:") or not selected_device[5:].isdigit():
        raise ValueError(f"invalid SAPIEN render device {selected_device!r}; expected cuda:N")

    selected_denoiser = str(
        denoiser or os.environ.get("ROBONANA_SAPIEN_DENOISER", "optix")
    ).strip().lower()
    if selected_denoiser not in {"none", "oidn", "optix"}:
        raise ValueError(f"unsupported SAPIEN denoiser {selected_denoiser!r}")

    if sapien_module is None:
        import sapien as sapien_module

    scene_class = sapien_module.wrapper.scene.Scene
    configured_device = getattr(scene_class, "_robonana_render_device", None)
    if configured_device is not None:
        if configured_device != selected_device:
            raise RuntimeError(
                f"SAPIEN was already configured for {configured_device}, not {selected_device}"
            )
        return selected_device, selected_denoiser

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
        original_set_denoiser(selected_denoiser)

    sapien_module.render.set_ray_tracing_denoiser = set_selected_denoiser
    print(
        f"[RoboNana SAPIEN] render_device={selected_device} denoiser={selected_denoiser}",
        flush=True,
    )
    return selected_device, selected_denoiser
