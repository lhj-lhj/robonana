from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import subprocess

import pytest

from robonana.sim.sapien_runtime import configure_sapien_runtime


def test_zero_throughput_dependency_patch_is_valid():
    # 中文：验证补丁格式及两个shader目标；实际编译/黑色目标回归必须在仿真GPU执行。
    # English: Check patch syntax/targets; compilation and black-target regression require a sim GPU.
    patch = Path(__file__).resolve().parents[1] / "patches/sapien/0002-terminate-zero-throughput-rays.patch"
    result = subprocess.run(["git", "apply", "--numstat", str(patch)],
                            check=True, capture_output=True, text=True)
    assert "vulkan_shader/rt/camera.rchit" in result.stdout
    assert "vulkan_shader/rt/camera.rgen" in result.stdout


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
    camera.entity.get_pose = lambda: SimpleNamespace(p=[1, 2, 3], q=[1, 0, 0, 0])
    camera.take_picture()

    assert camera.calls == 1
    output = capsys.readouterr().out
    assert "take_picture begin camera='head_camera'" in output
    assert "take_picture end camera='head_camera'" in output
    assert "camera_pose p=[1, 2, 3] q=[1, 0, 0, 0]" in output


@pytest.mark.parametrize("device", ["", "cuda", "cuda:x", "gpu:1"])
def test_configure_sapien_runtime_rejects_ambiguous_devices(device):
    module, _ = fake_sapien()
    with pytest.raises(ValueError):
        configure_sapien_runtime(device=device, sapien_module=module)
