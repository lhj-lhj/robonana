"""Collection entrypoint wiring is testable without a live simulator."""
import subprocess
import sys
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
