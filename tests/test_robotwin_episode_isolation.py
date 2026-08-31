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


def test_bootstrap_runs_one_episode_from_explicit_seed(tmp_path: Path) -> None:
    bootstrap = load_script("robotwin_eval_bootstrap_test", "scripts/robotwin_eval_bootstrap.py")
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
        "robotwin_eval_bootstrap_cameras", "scripts/robotwin_eval_bootstrap.py"
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
    isolated = load_script("robotwin_task_isolated_modes", "scripts/eval_robotwin_task_isolated.py")

    assert [(mode.name, mode.oidn_device) for mode in isolated.attempt_modes(2, True)] == [
        ("oidn_cuda_1", "cuda"),
        ("oidn_cuda_2", "cuda"),
        ("oidn_cpu_fallback", "cpu"),
    ]
    assert [mode.oidn_device for mode in isolated.attempt_modes(1, False)] == ["cuda"]


def test_cpu_fallback_is_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolated = load_script("robotwin_task_isolated_defaults", "scripts/eval_robotwin_task_isolated.py")
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
    isolated = load_script("robotwin_task_isolated_ledger", "scripts/eval_robotwin_task_isolated.py")
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
