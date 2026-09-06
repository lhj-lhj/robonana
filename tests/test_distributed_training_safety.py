import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("bad,micro", [("nan", 0), ("nan", 1), ("inf", 0), ("inf", 1)])
def test_real_accelerate_two_rank_nonfinite_aborts_before_step(bad, micro):
    if sys.platform == "win32":
        pytest.skip("distributed integration is validated on Linux/190")
    script = Path(__file__).resolve().parents[1] / "scripts/validate_mac_distributed_safety.py"
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
         str(script), "--bad", bad, "--micro", str(micro)],
        env={**os.environ, "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "PASS"' in result.stdout
