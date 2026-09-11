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


def test_isolated_runner_accepts_explicit_retry_seed(tmp_path,monkeypatch):
    runner=load_script('isolated_seed_override','scripts/internal/eval_robotwin_task_isolated.py')
    client=tmp_path/'client.sh';client.touch()
    monkeypatch.setenv('ROBONANA_EVAL_START_SEED','100038')
    monkeypatch.setattr(sys,'argv',['runner','--task-name','place_fan','--task-config','demo_clean',
        '--test-num','1','--output-dir',str(tmp_path/'out'),'--launch-client',str(client)])
    assert runner.parse_args().start_seed==100038


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


def test_attempt_modes_keep_oidn_enabled_and_make_cpu_fallback_explicit() -> None:
    isolated = load_script("robotwin_task_isolated_modes", "scripts/internal/eval_robotwin_task_isolated.py")

    assert [(mode.name, mode.oidn_device) for mode in isolated.attempt_modes(2, True)] == [
        ("oidn_cuda_1", "cuda"),
        ("oidn_cuda_2", "cuda"),
        ("oidn_cpu_fallback", "cpu"),
    ]
    assert [mode.oidn_device for mode in isolated.attempt_modes(1, False)] == ["cuda"]


def test_cpu_fallback_is_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolated = load_script("robotwin_task_isolated_defaults", "scripts/internal/eval_robotwin_task_isolated.py")
    launch_client = tmp_path / "launch_client.sh"
    launch_client.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_robotwin_task_isolated.py",
            "--task-name",
            "move_stapler_pad",
            "--task-config",
            "demo_clean",
            "--test-num",
            "1",
            "--output-dir",
            str(tmp_path / "output"),
            "--launch-client",
            str(launch_client),
        ],
    )

    assert isolated.parse_args().cpu_fallback is False


def test_ledger_requires_contiguous_episode_and_seed_chain(tmp_path: Path) -> None:
    isolated = load_script("robotwin_task_isolated_ledger", "scripts/internal/eval_robotwin_task_isolated.py")
    ledger = tmp_path / "episodes.jsonl"
    rows = [
        {
            "episode_index": 0,
            "start_seed": 100_000,
            "accepted_seed": 100_002,
            "next_seed": 100_003,
            "success": 1,
        },
        {
            "episode_index": 1,
            "start_seed": 100_003,
            "accepted_seed": 100_003,
            "next_seed": 100_004,
            "success": 0,
        },
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert isolated.read_ledger(ledger, 2, 100_000) == rows

    rows[1]["start_seed"] = 100_004
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="start seed"):
        isolated.read_ledger(ledger, 2, 100_000)


def test_swallowed_error_watchdog_reads_only_bounded_tail(tmp_path: Path) -> None:
    isolated = load_script("robotwin_task_isolated_watchdog", "scripts/internal/eval_robotwin_task_isolated.py")
    log = tmp_path / "episode.log"
    assert isolated.swallowed_error_count(log) == 0
    log.write_bytes(b"error occurs !\n" + b"x" * (2 * 1024 * 1024) + b"\nerror occurs !\n" * 3)
    assert isolated.swallowed_error_count(log) == 3
