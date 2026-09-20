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
    "run_multitask_mbrl.py",
    "run_robotwin_train.sh",
    "eval_robotwin_all_tasks_parallel.sh",
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
    "run_multitask_mbrl.py",
])
def test_relocated_help_entrypoint_imports_from_another_working_directory(tmp_path, relative):
    # Help exits before training, model loading, simulation or data writes.
    roots = [ROOT / p for p in ("src", "third_party/FACT", "third_party/flux2_official/src")]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, roots)),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(SCRIPTS / relative), "--help"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


def test_deleted_config_module_has_no_importers():
    for folder in (ROOT / 'src', SCRIPTS, ROOT / 'tests'):
        for path in folder.rglob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert node.module != 'robonana.configs.posttrain_config', path
                    if node.module == 'robonana.configs':
                        assert all(a.name != 'posttrain_config' for a in node.names), path
                elif isinstance(node, ast.Import):
                    assert all(a.name != 'robonana.configs.posttrain_config' for a in node.names), path


def test_config_paths_are_independent_of_working_directory(tmp_path, monkeypatch):
    from robonana.configs.evaluation import EvalOptions
    from robonana.configs.schema import load_options

    config = ROOT / 'configs' / 'eval.json'
    before = load_options(EvalOptions, config)
    monkeypatch.chdir(tmp_path)
    after = load_options(EvalOptions, config)
    assert before == after
    assert after.output == ROOT / 'outputs' / 'eval'
