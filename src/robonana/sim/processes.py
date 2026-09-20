"""评测子进程组清理；只终止当前调度器创建的子进程。"""
import os
import signal
import subprocess
from typing import Any


def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)

