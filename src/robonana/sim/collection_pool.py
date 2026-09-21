"""One synchronous RoboTwin slot per process, with atomic episode scheduling.

The supervisor owns the explicit wall-clock timeout. A second thread-level
timeout would abort valid cold resets on shared GPUs without cancelling them.
"""

from contextlib import closing
import json
from pathlib import Path
import sqlite3


def expert_planning_failure(error):
    """Only known expert planning failures may advance the candidate seed.

    RoboTwin's official expert loop skips these exceptions. Scope both their
    message and origin: a policy error, OOM or missing asset must still abort.
    """
    frames = []
    tb = error.__traceback__
    while tb is not None:
        frames.append((Path(tb.tb_frame.f_code.co_filename).name, tb.tb_frame.f_code.co_name))
        tb = tb.tb_next
    if isinstance(error, AssertionError) and str(error) == "target_pose cannot be None for move action.":
        return ("action.py", "__init__") in frames and ("_base_task.py", "grasp_actor") in frames
    if isinstance(error, TypeError) and str(error) == "'NoneType' object is not subscriptable":
        return ("click_alarmclock.py", "play_once") in frames
    if isinstance(error, IndexError) and str(error) == "list index out of range":
        return ("put_bottles_dustbin.py", "play_once") in frames
    from numpy.linalg import LinAlgError
    if isinstance(error, LinAlgError) and str(error) == "Eigenvalues did not converge":
        return ("transforms.py", "get_place_pose") in frames and ("_base_task.py", "place_actor") in frames
    return False


def initialize_policy_task_state(task):
    """Restore task-local success-check inputs normally set by play_once.

    Official evaluation reuses the expert's Python task object after reset.
    Our separate policy process must derive these same inputs from the initial
    scene, before moving anything. Never replay expert actions or consume RNG.
    """
    name = type(task).__name__
    if name not in ("open_laptop", "place_object_scale", "put_object_cabinet"):
        return
    from envs.utils import ArmTag, get_face_prod
    if name == "open_laptop":
        face = get_face_prod(task.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        task.arm_tag = ArmTag("left" if face > 0 else "right")
    else:
        task.arm_tag = ArmTag("right" if task.object.get_pose().p[0] > 0 else "left")
        if name == "put_object_cabinet":
            task.origin_z = task.object.get_pose().p[2]


def validate_jobs(jobs, workers):
    """Never silently repeat a seed or change its instruction during scheduling."""
    if workers < 1 or not jobs or workers > len(jobs):
        raise ValueError("workers must be between one and the number of jobs")
    seeds = [int(job["seed"]) for job in jobs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate episode seeds")
    if any(not str(job.get("instruction", "")).strip() for job in jobs):
        raise ValueError("each prevalidated seed needs its recorded instruction")


class EpisodeQueue:
    """Atomic FIFO claims shared by GPU-isolated workers, not a policy queue.

    Scheduling around the simulator's selected-seed reset. SQLite
    transactions only cover tiny metadata operations, never simulation/inference.
    Claimed work is NOT automatically retried: the supervisor fails the whole
    probe on worker death, avoiding duplicate trajectories/false completion.
    """
    def __init__(self, path):
        self.path = Path(path)

    def initialize(self, jobs):
        validate_jobs(jobs, 1)
        with self.path.open("xb"):
            pass
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            connection.execute("CREATE TABLE jobs (seed INTEGER PRIMARY KEY, ordinal INTEGER, "
                               "payload TEXT, status TEXT, owner TEXT, result TEXT)")
            connection.executemany("INSERT INTO jobs VALUES (?, ?, ?, 'pending', NULL, NULL)",
                [(int(job["seed"]), index, json.dumps(job)) for index, job in enumerate(jobs)])

    def claim(self, worker):
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT seed, payload FROM jobs WHERE status='pending' "
                                     "ORDER BY ordinal LIMIT 1").fetchone()
            if row is None:
                return None
            connection.execute("UPDATE jobs SET status='claimed', owner=? WHERE seed=?", (worker, row[0]))
            return json.loads(row[1])

    def complete(self, seed, worker, result):
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            cursor = connection.execute("UPDATE jobs SET status='done', result=? "
                "WHERE seed=? AND status='claimed' AND owner=?", (json.dumps(result), int(seed), worker))
            if cursor.rowcount != 1:
                raise RuntimeError("queue completion does not match a unique owned claim")

    def counts(self):
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            return dict(connection.execute("SELECT status, count(*) FROM jobs GROUP BY status"))


class RoboNanaSubEnv:
    """Thin synchronous task/client bridge for step/reset/close.

    Do NOT call RLinf gen_sparse_reward_data/chunk_step: those interfaces only
    expose chunk-end observations. Our adapter.eval saves every real transition
    and its true terminal observation before reset_model publishes the HDF5.
    An exception must propagate; an incomplete trajectory is not marked complete.
    Lifecycle reference: vector_env.py SubEnv.reset, periodic clear_cache=8.
    The custom bridge is necessary because stock SubEnv.step invokes the RLinf
    reward/action API rather than our unchanged per-control-step client/writer.
    """
    def __init__(self, task, model, jobs, task_args, adapter, clear_cache_freq=8,
                 audit_actions=False, defer_publish=False):
        if clear_cache_freq < 1:
            raise ValueError("clear_cache_freq must be positive")
        self.task, self.model, self.adapter = task, model, adapter
        self.jobs = {int(job["seed"]): job for job in jobs}
        self.args = task_args
        self.clear_cache_freq = clear_cache_freq
        self.completed = 0
        self.active = False
        self.done = True
        self.audit_actions = audit_actions
        self.defer_publish = defer_publish

    def reset(self, env_seed=None):
        if not self.done:
            raise RuntimeError("cannot reset an incomplete episode")
        job = self.jobs[int(env_seed)]  # No silent replacement of invalid seeds.
        if self.active:
            self.close(clear_cache=self.completed % self.clear_cache_freq == 0)
        self.task.setup_demo(now_ep_num=0, seed=int(env_seed), is_test=True, **self.args)
        self.active = True
        initialize_policy_task_state(self.task)
        self.task.set_instruction(instruction=job["instruction"])
        self.adapter.reset_model(self.model)
        self.done = False
        self.seed = int(env_seed)
        self.action_trace = []
        self.rgb_steps = 0

    def step(self, _unused):
        if self.done:
            raise RuntimeError("completed environment must be reset before stepping")
        observation = self.task.get_obs()
        step = int(self.task.take_action_cnt)
        if self.audit_actions and not observation.get('_fact_light_obs', False):
            self.rgb_steps += 1
        self.adapter.eval(self.task, self.model, observation)
        if self.audit_actions:
            self.action_trace.append(self.model.planned_actions[
                step % self.model.execute_actions_per_plan].copy())
        success = bool(self.task.eval_success)
        truncated = int(self.task.take_action_cnt) >= int(self.task.step_lim) and not success
        self.done = success or truncated
        if self.done:
            # Existing writer publishes true terminal obs before reset can destroy it.
            if not self.defer_publish:
                self.adapter.reset_model(self.model)
            self.completed += 1
        return {"obs": None, "reward": None, "terminated": success,
                "truncated": truncated, "info": {"seed": self.seed,
                "success": success, "steps": int(self.task.take_action_cnt)}}

    def close(self, clear_cache=True):
        if self.active:
            self.task.close_env(clear_cache=clear_cache)
            self.active = False

