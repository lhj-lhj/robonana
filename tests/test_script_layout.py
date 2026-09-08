"""中文：防止脚本分类后路径失效。 English: guard relocated script entry points."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PUBLIC = {
    "run_robotwin_train.sh", "run_hanging_mug_mac_round.sh",
    "collect_prepare_robotwin_rollouts.sh", "eval_robotwin_all_tasks_parallel.sh",
    "prepare_robotwin_rollouts.py", "report_selected_world_eval.py",
}


def test_only_public_entry_points_are_at_scripts_root():
    assert {p.name for p in SCRIPTS.iterdir() if p.suffix in {".py", ".sh"}} == PUBLIC


def test_every_script_has_bilingual_role_and_invocation_comments():
    for path in SCRIPTS.rglob("*"):
        if path.suffix not in {".py", ".sh"}:
            continue
        header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:6])
        assert "# 中文：" in header, path
        assert "# English:" in header, path
        assert "# 调用 / Invocation:" in header, path


def test_relocated_python_root_expressions_still_point_to_repository():
    checked = 0
    for path in SCRIPTS.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id in {"ROOT", "REPO_ROOT", "repo"}
                for target in node.targets
            ):
                continue
            if "__file__" not in ast.unparse(node.value):
                continue
            value = eval(compile(ast.Expression(node.value), str(path), "eval"),
                         {"Path": Path, "__file__": str(path)})
            assert value == ROOT, path
            checked += 1
    assert checked >= 6


@pytest.mark.skipif(os.name != "posix", reason="Bash syntax is validated on 190")
def test_all_shell_scripts_parse():
    for path in SCRIPTS.rglob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)


@pytest.mark.skipif(os.name != "posix", reason="Executable permissions are validated on 190")
def test_moved_executable_entrypoints_keep_their_permissions():
    for relative in ("env/robotwin_eval_python.sh", "env/install_sapien_oidn_blackwell.sh",
                     "services/inference_server_robotwin_xpolicylab.py"):
        assert os.access(SCRIPTS / relative, os.X_OK), relative


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Dependency environment is on 190")
@pytest.mark.parametrize("relative", [
    "services/inference_server_robotwin.py",
    "services/inference_server_robotwin_batched.py",
    "services/inference_server_robotwin_xpolicylab.py",
    "prepare_robotwin_rollouts.py",
    "diagnostics/verify_image_pipeline.py",
    "internal/collect_robotwin_pool_worker.py",
    "internal/train_robotwin.py",
])
def test_relocated_help_entrypoint_imports_from_another_working_directory(tmp_path, relative):
    # Help exits before training, model loading, simulation or data writes.
    roots = [ROOT / p for p in ("src", "third_party/FACT", "third_party/flux2/src", "third_party/flux2_official/src")]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, roots)),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(SCRIPTS / relative), "--help"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
