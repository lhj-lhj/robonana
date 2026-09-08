from types import SimpleNamespace

import pytest

from concurrent.futures import ThreadPoolExecutor

from robonana.sim.collection_pool import EpisodeQueue, RoboNanaSubEnv, validate_jobs


def test_valid_seed_list():
    jobs = [{"seed": i, "instruction": "hang mug"} for i in range(4)]
    assert validate_jobs(jobs, 2) is None


@pytest.mark.parametrize("jobs,workers", [([], 1), ([{"seed": 1}], 1),
    ([{"seed": 1, "instruction": "x"}] * 2, 2)])
def test_invalid_jobs_rejected(jobs, workers):
    with pytest.raises(ValueError):
        validate_jobs(jobs, workers)


def test_dynamic_queue_claims_once_and_finishes(tmp_path):
    queue = EpisodeQueue(tmp_path / "queue.sqlite")
    queue.initialize([{"seed": seed, "instruction": "hang"} for seed in range(24)])
    def worker(owner):
        seeds = []
        while (job := queue.claim(owner)) is not None:
            seeds.append(job["seed"])
            queue.complete(job["seed"], owner, {"success": False})
        return seeds
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, ["0", "1", "2", "3"]))
    assert sorted(seed for group in results for seed in group) == list(range(24))
    assert queue.counts() == {"done": 24}


def test_queue_rejects_wrong_owner_and_duplicate_completion(tmp_path):
    queue = EpisodeQueue(tmp_path / "queue.sqlite")
    queue.initialize([{"seed": 1, "instruction": "hang"}])
    queue.claim("worker")
    with pytest.raises(RuntimeError):
        queue.complete(1, "intruder", {})
    assert queue.counts() == {"claimed": 1}
    queue.complete(1, "worker", {})
    with pytest.raises(RuntimeError):
        queue.complete(1, "worker", {})
    with pytest.raises(FileExistsError):
        queue.initialize([{"seed": 2, "instruction": "hang"}])


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
