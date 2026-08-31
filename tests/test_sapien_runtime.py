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


class FakeCamera:
    def __init__(self):
        self.entity = SimpleNamespace(name="head_camera")
        self.calls = 0

    def take_picture(self):
        self.calls += 1


def fake_sapien():
    if hasattr(FakeScene, "_robonana_render_device"):
        delattr(FakeScene, "_robonana_render_device")
    denoisers = []
    module = SimpleNamespace()
    module.wrapper = SimpleNamespace(scene=SimpleNamespace(Scene=FakeScene))
    module.pysapien = SimpleNamespace(
        physx=SimpleNamespace(PhysxCpuSystem=lambda: "cpu"),
        render=SimpleNamespace(
            RenderSystem=FakeRenderSystem,
            RenderCameraComponent=FakeCamera,
        ),
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


def test_configure_sapien_runtime_can_trace_camera_calls(monkeypatch, capsys):
    module, _ = fake_sapien()
    monkeypatch.setenv("ROBONANA_SAPIEN_TRACE_CAMERAS", "1")

    configure_sapien_runtime(device="cuda:0", sapien_module=module)
    camera = module.pysapien.render.RenderCameraComponent()
    camera.take_picture()

    assert camera.calls == 1
    output = capsys.readouterr().out
    assert "take_picture begin camera='head_camera'" in output
    assert "take_picture end camera='head_camera'" in output


@pytest.mark.parametrize("device", ["", "cuda", "cuda:x", "gpu:1"])
def test_configure_sapien_runtime_rejects_ambiguous_devices(device):
    module, _ = fake_sapien()
    with pytest.raises(ValueError):
        configure_sapien_runtime(device=device, sapien_module=module)
