"""Opt-in probe only: defer action render synchronization, never observations.

Reference: FACT evaluation/robotwin/model2robotwin_interface.py,
_take_action_low_frequency_rgb in the server's third_party/FACT dependency.
The wrapped upstream simulator implementation is Base_Task.take_action/get_obs:
https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/_base_task.py
Unlike that adapter, this does not reduce observation frequency. The production
client does not import/install it. Validate in SAPIEN before enabling anywhere.
"""

from contextlib import contextmanager


@contextmanager
def defer_action_render_sync(task, *, enabled):
    """Wrap one environment instance and restore methods even on exceptions.

    RoboTwin take_action calls get_obs internally on early success. That nested
    observation MUST run the real render sync, otherwise its terminal RGB is
    stale. Dynamic lighting is rejected because _update_render consumes RNG and
    mutates lighting; dropping those calls would change the experiment.
    """
    if enabled and (
        getattr(task, "crazy_random_light", False)
        or getattr(task, "render_freq", 0)
        or getattr(task, "eval_video_path", None) is not None
    ):
        raise ValueError("render probe requires static lighting, no viewer/video")
    names = ("_update_render", "get_obs", "take_action")
    saved = {name: task.__dict__.get(name) for name in names}
    existed = {name: name in task.__dict__ for name in names}
    update, observe, act = (getattr(task, name) for name in names)
    depth = {"action": 0, "observation": 0}
    counts = {"sync": 0, "skipped": 0}

    def wrapped_update(*args, **kwargs):
        if enabled and depth["action"] and not depth["observation"]:
            counts["skipped"] += 1
            # Match FACT's wrist-pose updates, suppress only scene render sync.
            task.cameras.update_wrist_camera(
                task.robot.left_camera.get_pose(), task.robot.right_camera.get_pose()
            )
            return None
        counts["sync"] += 1
        return update(*args, **kwargs)

    def wrapped_observe(*args, **kwargs):
        depth["observation"] += 1
        try:
            return observe(*args, **kwargs)
        finally:
            depth["observation"] -= 1

    def wrapped_act(*args, **kwargs):
        depth["action"] += 1
        try:
            return act(*args, **kwargs)
        finally:
            depth["action"] -= 1

    task._update_render = wrapped_update
    task.get_obs = wrapped_observe
    task.take_action = wrapped_act
    try:
        yield counts
    finally:
        for name in names:
            if existed[name]:
                setattr(task, name, saved[name])
            else:
                delattr(task, name)
