from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for candidate in (
    os.environ.get("FACT_SRC"),
    os.environ.get("FLUX2_SRC"),
    ROOT / "third_party" / "FACT",
    ROOT / "third_party" / "flux2" / "src",
    ROOT / "third_party" / "flux2_official" / "src",
    ROOT.parent / "FACT",
    ROOT.parent / "flux2" / "src",
):
    if candidate and Path(candidate).exists():
        sys.path.insert(0, str(candidate))
