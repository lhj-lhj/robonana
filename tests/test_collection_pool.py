from types import SimpleNamespace

import pytest

from robonana.sim.collection_pool import RoboNanaSubEnv, partition_jobs


def test_disjoint_seed_assignment():
    jobs = [{"seed": i, "instruction": "hang mug"} for i in range(4)]
    assert partition_jobs(jobs, 2) == [jobs[::2], jobs[1::2]]


@pytest.mark.parametrize("jobs,workers", [([], 1), ([{"seed": 1}], 1),
    ([{"seed": 1, "instruction": "x"}] * 2, 2)])
def test_invalid_jobs_rejected(jobs, workers):
    with pytest.raises(ValueError):
        partition_jobs(jobs, workers)


@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize("previous,clear", [(0, False), (7, True)])
def test_per_step_recording_and_terminal_publish_before_close(success, previous, clear):
    events = []
    task = SimpleNamespace(take_action_cnt=0, step_lim=3, eval_success=False)
    def setup(**kw):
        task.take_action_cnt = 0
        task.eval_success = False
        events.append(("setup", kw["seed"]))
    task.setup_demo = setup
    task.set_instruction = lambda **kw: None
    task.get_obs = lambda: events.append("obs")
    task.close_env = lambda **kw: events.append(("close", kw["clear_cache"]))
    def step(task, model, obs):
        task.take_action_cnt += 1
        task.eval_success = success
        events.append("write_transition_and_terminal_if_done")
    adapter = SimpleNamespace(eval=step, reset_model=lambda _: events.append("publish"))
    slot = RoboNanaSubEnv(task, None, [{"seed": 42, "instruction": "hang"}], {}, adapter)
    slot.completed = previous
    slot.reset(42)
    while not slot.done:
        result = slot.step(None)["info"]
    assert result["steps"] == (1 if success else 3)
    assert events.count("obs") == result["steps"]
    assert events[-1] == "publish"
    slot.reset(42)
    assert events[-3:] == [("close", clear), ("setup", 42), "publish"]
