from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_exception_trace_reports_live_seed_without_tracing_planner(monkeypatch, capsys):
    bootstrap = load_script("bootstrap_exception_context", "scripts/env/robotwin_eval_bootstrap.py")
    hooks = []
    monkeypatch.setenv("ROBONANA_EVAL_DEBUG", "1")
    monkeypatch.setattr(sys, "settrace", hooks.append)
    bootstrap._install_exception_trace()
    trace = hooks[0]
    frame = SimpleNamespace(
        f_code=SimpleNamespace(co_filename="/robotwin/script/eval_policy.py", co_name="eval_policy"),
        f_locals={"now_seed": 100042, "TASK_ENV": SimpleNamespace(take_action_cnt=96)},
    )
    assert trace(frame, "exception", (RuntimeError, RuntimeError("device lost"), None)) is trace
    assert "'seed': 100042, 'control_step': 96" in capsys.readouterr().err
    frame.f_code.co_filename = "/planner/planner.py"
    assert trace(frame, "call", None) is None


def test_bootstrap_runs_one_episode_from_explicit_seed(tmp_path: Path) -> None:
    bootstrap = load_script("robotwin_eval_bootstrap_test", "scripts/env/robotwin_eval_bootstrap.py")
    entrypoint = tmp_path / "eval_policy.py"
    entrypoint.write_text(
        """
def parse_args_and_config():
    return {"test_num": 50}

def eval_policy(task, env, args, model, st_seed, test_num=100):
    assert test_num == 1
    return st_seed + 4, 1

def main(usr_args):
    assert usr_args["test_num"] == 1
    eval_policy(None, None, None, None, 999, test_num=usr_args["test_num"])
""",
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.json"

    bootstrap._run_one_isolated_episode(entrypoint, 100_007, metadata)

    assert json.loads(metadata.read_text(encoding="utf-8")) == {
        "accepted_seed": 100_010,
        "next_seed": 100_011,
        "start_seed": 100_007,
        "success": 1,
    }


def test_bootstrap_retains_only_policy_static_cameras() -> None:
    bootstrap = load_script(
        "robotwin_eval_bootstrap_cameras", "scripts/env/robotwin_eval_bootstrap.py"
    )
    camera_bundle = SimpleNamespace(
        static_camera_name=["head_camera", "front_camera"],
        static_camera_list=["head", "front"],
        static_camera_config=["head_config", "front_config"],
        head_camera_id=0,
    )

    removed = bootstrap._retain_static_cameras(camera_bundle, ("head_camera",))

    assert removed == ("front_camera",)
    assert camera_bundle.static_camera_name == ["head_camera"]
    assert camera_bundle.static_camera_list == ["head"]
    assert camera_bundle.static_camera_config == ["head_config"]
    assert camera_bundle.head_camera_id == 0


