"""The one maintained state/action coordinate system: Stage-1 statistics A.

Dataset location must never select normalization. In particular, replay is
not allowed to fit its own statistics or inherit the older HDF5 statistics B.
No torch dependency: launch/config tools can validate paths before loading models.
"""
import json
from pathlib import Path

A_STATS_PATH = Path("/workspace/datasets/fact-robotwin-v2/RoboTwin/robonana_norm_stats.json")


def require_a_stats_path(path=None) -> Path:
    canonical = A_STATS_PATH.expanduser().resolve()
    if path is not None and Path(path).expanduser().resolve() != canonical:
        raise ValueError(f"Only Stage-1 normalization A is supported: {canonical}; got {path}")
    return A_STATS_PATH.expanduser()


def load_a_stats(path=None) -> dict:
    return json.loads(require_a_stats_path(path).read_text(encoding="utf-8"))
