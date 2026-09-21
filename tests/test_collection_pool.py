from types import SimpleNamespace

import pytest

from concurrent.futures import ThreadPoolExecutor

from robonana.sim.collection_pool import EpisodeQueue, RoboNanaSubEnv, validate_jobs


@pytest.mark.parametrize('filename,function,error,expected', [
    ('click_alarmclock.py', 'play_once', TypeError("'NoneType' object is not subscriptable"), True),
    ('put_bottles_dustbin.py', 'play_once', IndexError('list index out of range'), True),
    ('client.py', 'play_once', TypeError("'NoneType' object is not subscriptable"), False),
    ('click_alarmclock.py', 'setup_demo', TypeError("'NoneType' object is not subscriptable"), False),
    ('click_alarmclock.py', 'play_once', RuntimeError('CUDA out of memory'), False),
    ('click_alarmclock.py', 'play_once', FileNotFoundError('asset missing'), False),
])
def test_expert_rejection_checks_origin_not_just_error_text(filename, function, error, expected):
    from robonana.sim.collection_pool import expert_planning_failure
    scope = {'error': error}
    exec(compile(f'def {function}():\n    raise error\n', filename, 'exec'), scope)
    try:
        scope[function]()
    except Exception as caught:
        assert expert_planning_failure(caught) is expected


def test_grasp_rejection_requires_planner_and_action_frames():
    from robonana.sim.collection_pool import expert_planning_failure
    scope = {}
    exec(compile('def __init__():\n    raise AssertionError("target_pose cannot be None for move action.")',
                 'action.py', 'exec'), scope)
    exec(compile('def grasp_actor():\n    __init__()', '_base_task.py', 'exec'), scope)
    for function, expected in [('__init__', False), ('grasp_actor', True)]:
        try:
            scope[function]()
        except AssertionError as caught:
            assert expert_planning_failure(caught) is expected


@pytest.mark.parametrize('name,side', [('open_laptop','left'), ('place_object_scale','right'),
                                    ('put_object_cabinet','right')])
def test_success_check_state_is_reinitialized_from_current_scene(monkeypatch, name, side):
    import sys
    from robonana.sim.collection_pool import initialize_policy_task_state
    monkeypatch.setitem(sys.modules, 'envs.utils', SimpleNamespace(ArmTag=str, get_face_prod=lambda *a:1))
    task = type(name, (), {})()
    pose = SimpleNamespace(p=[0.2, 0., 0.81], q=[1., 0., 0., 0.])
    task.object = task.laptop = SimpleNamespace(get_pose=lambda:pose)
    task.arm_tag, task.origin_z = 'stale', -1
    initialize_policy_task_state(task)
    assert task.arm_tag == side
    if name == 'put_object_cabinet':
        assert task.origin_z == .81
        pose.p[2] = .85
        initialize_policy_task_state(task)
        assert task.origin_z == .85


def test_valid_seed_list():
    jobs = [{"seed": i, "instruction": "hang mug"} for i in range(4)]
    assert validate_jobs(jobs, 2) is None


def test_replay_defers_publication_until_audit():
    import numpy as np
    publications=[]
    task=SimpleNamespace(take_action_cnt=0,step_lim=1,eval_success=False,
                         get_obs=lambda:{'_fact_light_obs':True},
                         setup_demo=lambda **kw:None,set_instruction=lambda **kw:None)
    model=SimpleNamespace(planned_actions=np.ones((48,14)),execute_actions_per_plan=48)
    def advance(task,model,obs):task.take_action_cnt+=1
    adapter=SimpleNamespace(eval=advance,reset_model=lambda m:publications.append(True))
    slot=RoboNanaSubEnv(task,model,[{'seed':123,'instruction':'hang'}],{},adapter,
                       audit_actions=True,defer_publish=True)
    slot.reset(123)
    publications.clear()
    result=slot.step(None)
    assert result['truncated'] and slot.done
    assert len(slot.action_trace)==1 and slot.rgb_steps==0
    assert not publications


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
