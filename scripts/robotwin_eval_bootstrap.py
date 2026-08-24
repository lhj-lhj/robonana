#!/usr/bin/env python3
"""Configure SAPIEN before executing RoboTwin's unmodified eval entrypoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from robonana.sim import configure_sapien_runtime


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: robotwin_eval_bootstrap.py ENTRYPOINT [ARGS...]")
    configure_sapien_runtime()
    entrypoint = Path(sys.argv[1]).resolve()
    if not entrypoint.is_file():
        raise FileNotFoundError(f"RoboTwin entrypoint does not exist: {entrypoint}")
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
