from types import SimpleNamespace

import pytest

from robonana.sim.render_sync_probe import defer_action_render_sync


class Task:
    crazy_random_light = False
    render_freq = 0
    eval_video_path = None

    def __init__(self):
        self.renders = 0
        camera = SimpleNamespace(get_pose=lambda: None)
        self.robot = SimpleNamespace(left_camera=camera, right_camera=camera)
        self.cameras = SimpleNamespace(update_wrist_camera=lambda *args: None)

    def _update_render(self):
        self.renders += 1

    def get_obs(self):
        self._update_render()
        return self.renders

    def take_action(self, *, fail=False):
        self._update_render()
        self._update_render()
        if fail:
            raise RuntimeError("test")
        return self.get_obs()  # Early-success terminal observation.


@pytest.mark.parametrize("enabled,expected", [(False, 3), (True, 1)])
def test_terminal_observation_is_always_synchronized(enabled, expected):
    task = Task()
    with defer_action_render_sync(task, enabled=enabled) as counts:
        assert task.take_action() == expected
        assert counts["skipped"] == (2 if enabled else 0)
        task.get_obs()
        assert task.renders == expected + 1
    assert "get_obs" not in task.__dict__
    assert "take_action" not in task.__dict__
    task.take_action()
    assert task.renders == expected + 4


def test_exception_restores_instance_methods():
    task = Task()
    with pytest.raises(RuntimeError), defer_action_render_sync(task, enabled=True):
        task.take_action(fail=True)
    assert "_update_render" not in task.__dict__
    task.take_action()
    assert task.renders == 3


@pytest.mark.parametrize("name,value", [
    ("crazy_random_light", True), ("render_freq", 1), ("eval_video_path", "video")
])
def test_unsafe_modes_rejected(name, value):
    task = Task()
    setattr(task, name, value)
    with pytest.raises(ValueError), defer_action_render_sync(task, enabled=True):
        pass
