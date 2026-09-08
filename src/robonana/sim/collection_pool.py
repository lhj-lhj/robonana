"""Collection scheduling only; no policy, reward or world-model changes.

Directly loads the official VectorEnv; never imports the RLinf trainer:
https://github.com/RoboTwin-Platform/RoboTwin/blob/RLinf_support/robotwin/envs/vector_env.py
  SubEnv.reset: persistent task, selected-env reset, periodic clear_cache.
https://github.com/RLinf/RLinf/blob/main/rlinf/envs/robotwin/seed_utils.py
  partition_success_seeds: prevalidated, disjoint seed assignment.

We use one official VectorEnv slot per GPU-bound process:
the existing RoboTwin uses process-global NumPy RNG and SAPIEN configuration.
This preserves the deployed simulator, camera and action execution semantics.
"""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import subprocess

RLINF_ROBOTWIN_COMMIT = "0008ae6800df9f75fc8de7098bacb01735fd8fd2"


def load_vector_env(checkout):
    """Load only the pinned environment library, using CURRENT RoboTwin imports.

    Do not add the RLinf_support checkout to sys.path: that would replace the
    deployed envs._base_task and task reward/control code as an accidental side
    effect. Only vector_env.py is loaded, and all envs imports resolve to the
    current simulator. This dependency is source, not an edited local copy.
    """
    checkout = Path(checkout)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if revision != RLINF_ROBOTWIN_COMMIT:
        raise RuntimeError(f"RLinf VectorEnv revision mismatch: {revision}")
    relative = "robotwin/envs/vector_env.py"
    original = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=checkout)
    if (checkout / relative).read_bytes() != original:
        raise RuntimeError("official VectorEnv dependency has local modifications")
    spec = importlib.util.spec_from_file_location("_robonana_official_vector_env", checkout / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VectorEnv


def partition_jobs(jobs, workers):
    """Never silently repeat a seed or change its instruction during sharding."""
    if workers < 1 or not jobs or workers > len(jobs):
        raise ValueError("workers must be between one and the number of jobs")
    seeds = [int(job["seed"]) for job in jobs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate episode seeds")
    if any(not str(job.get("instruction", "")).strip() for job in jobs):
        raise ValueError("each prevalidated seed needs its recorded instruction")
    return [jobs[index::workers] for index in range(workers)]


class RoboNanaSubEnv:
    """Thin task/client bridge for official VectorEnv.step/reset/close.

    Do NOT call RLinf gen_sparse_reward_data/chunk_step: those interfaces only
    expose chunk-end observations. Our adapter.eval saves every real transition
    and its true terminal observation before reset_model publishes the HDF5.
    An exception must propagate; an incomplete trajectory is not marked complete.
    Lifecycle reference: vector_env.py SubEnv.reset, periodic clear_cache=8.
    The custom bridge is necessary because stock SubEnv.step invokes the RLinf
    reward/action API rather than our unchanged per-control-step client/writer.
    """
    def __init__(self, task, model, jobs, task_args, adapter, clear_cache_freq=8):
        if clear_cache_freq < 1:
            raise ValueError("clear_cache_freq must be positive")
        self.task, self.model, self.adapter = task, model, adapter
        self.jobs = {int(job["seed"]): job for job in jobs}
        self.args = task_args
        self.clear_cache_freq = clear_cache_freq
        self.completed = 0
        self.active = False
        self.done = True

    def reset(self, env_seed=None):
        if not self.done:
            raise RuntimeError("cannot reset an incomplete episode")
        job = self.jobs[int(env_seed)]  # No silent replacement of invalid seeds.
        if self.active:
            self.close(clear_cache=self.completed % self.clear_cache_freq == 0)
        self.task.setup_demo(now_ep_num=0, seed=int(env_seed), is_test=True, **self.args)
        self.active = True
        self.task.set_instruction(instruction=job["instruction"])
        self.adapter.reset_model(self.model)
        self.done = False
        self.seed = int(env_seed)

    def step(self, _unused):
        if self.done:
            raise RuntimeError("completed environment must be reset before stepping")
        observation = self.task.get_obs()
        self.adapter.eval(self.task, self.model, observation)
        success = bool(self.task.eval_success)
        truncated = int(self.task.take_action_cnt) >= int(self.task.step_lim) and not success
        self.done = success or truncated
        if self.done:
            # Existing writer publishes true terminal obs before reset can destroy it.
            self.adapter.reset_model(self.model)
            self.completed += 1
        return {"obs": None, "reward": None, "terminated": success,
                "truncated": truncated, "info": {"seed": self.seed,
                "success": success, "steps": int(self.task.take_action_cnt)}}

    def close(self, clear_cache=True):
        if self.active:
            self.task.close_env(clear_cache=clear_cache)
            self.active = False


def make_vector_env(slot, vector_class):
    """Reuse upstream scheduling verbatim, inject our already-configured slot.

    Deliberately bypass VectorEnv.__init__ asset/task construction, which changes
    embodiment/planner/config assumptions. Native step/reset/transform/close
    methods and native timeout handling are used without copying or patching.
    """
    vector = vector_class.__new__(vector_class)
    vector.n_envs = 1
    vector.envs = [slot]
    vector.env_thread_pool = ThreadPoolExecutor(max_workers=1)
    return vector
