from __future__ import annotations

from types import SimpleNamespace

import pytest

from robonana.sim.sapien_runtime import configure_sapien_runtime


class FakeScene:
    def __init__(self, systems=None):
        self.systems = systems


class FakeRenderSystem:
    def __init__(self, device):
        self.device = device


def fake_sapien():
    denoisers = []
    module = SimpleNamespace()
    module.wrapper = SimpleNamespace(scene=SimpleNamespace(Scene=FakeScene))
    module.pysapien = SimpleNamespace(
        physx=SimpleNamespace(PhysxCpuSystem=lambda: "cpu"),
        render=SimpleNamespace(RenderSystem=FakeRenderSystem),
    )
    module.render = SimpleNamespace(
        set_ray_tracing_denoiser=lambda value: denoisers.append(value)
    )
    return module, denoisers


def test_configure_sapien_runtime_selects_physical_gpu_and_oidn():
    module, denoisers = fake_sapien()

    assert configure_sapien_runtime(
        device="cuda:6", sapien_module=module
    ) == ("cuda:6", "oidn")
    scene = module.wrapper.scene.Scene()
    module.render.set_ray_tracing_denoiser("ignored-by-robonana")

    assert scene.systems[0] == "cpu"
    assert scene.systems[1].device == "cuda:6"
    assert denoisers == ["oidn"]


@pytest.mark.parametrize("device", ["", "cuda", "cuda:x", "gpu:1"])
def test_configure_sapien_runtime_rejects_ambiguous_devices(device):
    module, _ = fake_sapien()
    with pytest.raises(ValueError):
        configure_sapien_runtime(device=device, sapien_module=module)
