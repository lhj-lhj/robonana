"""Collection entrypoint wiring is testable without a live simulator."""
import subprocess
import sys
import importlib.util
import pytest
from pathlib import Path


def test_collection_supervisor_and_seed_preflight_help(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for relative, required in (
        ('scripts/diagnostics/benchmark_robotwin_collection_pool.py', ('--jobs-json', '--inference-mode')),
        ('scripts/internal/collect_robotwin_pool_worker.py', ('--prepare-seeds', '--seed-start')),
    ):
        result = subprocess.run([sys.executable, str(root / relative), '--help'],
                                cwd=tmp_path, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert all(flag in result.stdout for flag in required)


def test_scene_manifest_rejects_duplicates_counts_and_unvalidated_sources():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('manifest', root / 'scripts/data/publish_robotwin_scene_manifest.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = dict(task_name='hanging_mug', task_config='demo_clean', expert_validated=True,
                  jobs=[dict(seed=3, instruction='hang mug'), dict(seed=1, instruction='hang mug')])
    result = module.merge_manifests([source], 2)
    assert [j['seed'] for j in result['jobs']] == [1, 3]
    for payloads, count in (([source, source], 4), ([source], 100),
                            ([dict(source, expert_validated=False)], 2)):
        with pytest.raises(ValueError):
            module.merge_manifests(payloads, count)
