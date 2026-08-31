#!/usr/bin/env python3
"""Configure SAPIEN before executing RoboTwin's unmodified eval entrypoint."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

from robonana.sim import configure_sapien_runtime


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _retain_static_cameras(camera_bundle: Any, allowed_names: tuple[str, ...]) -> tuple[str, ...]:
    names = list(camera_bundle.static_camera_name)
    cameras = list(camera_bundle.static_camera_list)
    configs = list(camera_bundle.static_camera_config)
    if not (len(names) == len(cameras) == len(configs)):
        raise RuntimeError("RoboTwin static camera lists have inconsistent lengths")
    missing = [name for name in allowed_names if name not in names]
    if missing:
        raise RuntimeError(f"requested RoboTwin static cameras are missing: {missing}")
    keep = [index for index, name in enumerate(names) if name in allowed_names]
    removed = tuple(name for name in names if name not in allowed_names)
    camera_bundle.static_camera_name = [names[index] for index in keep]
    camera_bundle.static_camera_list = [cameras[index] for index in keep]
    camera_bundle.static_camera_config = [configs[index] for index in keep]
    camera_bundle.head_camera_id = (
        camera_bundle.static_camera_name.index("head_camera")
        if "head_camera" in camera_bundle.static_camera_name
        else None
    )
    return removed


def _install_static_camera_filter() -> None:
    raw = os.environ.get("ROBONANA_ROBOTWIN_STATIC_CAMERAS", "").strip()
    if not raw:
        return
    allowed_names = tuple(dict.fromkeys(name.strip() for name in raw.split(",") if name.strip()))
    if not allowed_names:
        raise RuntimeError("ROBONANA_ROBOTWIN_STATIC_CAMERAS resolved to an empty list")

    from envs.camera.camera import Camera

    if getattr(Camera, "_robonana_static_camera_filter", None) == allowed_names:
        return
    if hasattr(Camera, "_robonana_static_camera_filter"):
        raise RuntimeError("RoboTwin static camera filter was already configured differently")
    original_load_camera = Camera.load_camera

    def load_camera(camera_bundle, scene):
        result = original_load_camera(camera_bundle, scene)
        removed = _retain_static_cameras(camera_bundle, allowed_names)
        print(
            f"[RoboNana SAPIEN] static_cameras={allowed_names} removed={removed}",
            flush=True,
        )
        return result

    Camera.load_camera = load_camera
    Camera._robonana_static_camera_filter = allowed_names


def _run_one_isolated_episode(entrypoint: Path, start_seed: int, metadata_path: Path) -> None:
    """Run exactly one accepted RoboTwin episode from an explicit seed.

    RoboTwin's evaluator may reject unstable expert seeds before accepting one.
    The wrapped return value therefore records the *next* seed, allowing a fresh
    process to resume the canonical deterministic sequence without duplicates or
    silently skipped policy episodes.
    """

    namespace = runpy.run_path(str(entrypoint), run_name="_robonana_robotwin_eval")
    _install_static_camera_filter()
    parse_args = namespace.get("parse_args_and_config")
    main = namespace.get("main")
    original_eval = namespace.get("eval_policy")
    if not callable(parse_args) or not callable(main) or not callable(original_eval):
        raise RuntimeError(
            "RoboTwin eval entrypoint must define parse_args_and_config, main, and eval_policy"
        )

    def eval_one(*args, **kwargs):
        positional = list(args)
        if len(positional) < 5:
            raise RuntimeError("unexpected RoboTwin eval_policy signature")
        positional[4] = start_seed
        kwargs["test_num"] = 1
        next_seed, success = original_eval(*positional, **kwargs)
        next_seed = int(next_seed)
        success = int(success)
        if next_seed <= start_seed:
            raise RuntimeError(
                f"RoboTwin returned non-advancing seed {next_seed} from {start_seed}"
            )
        if success not in (0, 1):
            raise RuntimeError(f"single-episode success must be 0 or 1, got {success}")
        _atomic_write_json(
            metadata_path,
            {
                "accepted_seed": next_seed - 1,
                "next_seed": next_seed,
                "start_seed": start_seed,
                "success": success,
            },
        )
        return next_seed, success

    # Functions produced by runpy resolve globals through their own globals dict.
    main.__globals__["eval_policy"] = eval_one
    usr_args = parse_args()
    usr_args["test_num"] = 1
    main(usr_args)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: robotwin_eval_bootstrap.py ENTRYPOINT [ARGS...]")
    configure_sapien_runtime()
    entrypoint = Path(sys.argv[1]).resolve()
    if not entrypoint.is_file():
        raise FileNotFoundError(f"RoboTwin entrypoint does not exist: {entrypoint}")
    # Match ``python path/to/script.py`` so sibling imports such as
    # RoboTwin's ``from test_render import Sapien_TEST`` continue to work.
    sys.path.insert(0, str(entrypoint.parent))
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    isolated_seed = os.environ.get("ROBONANA_EVAL_START_SEED")
    metadata = os.environ.get("ROBONANA_EVAL_EPISODE_METADATA")
    if (isolated_seed is None) != (metadata is None):
        raise RuntimeError(
            "ROBONANA_EVAL_START_SEED and ROBONANA_EVAL_EPISODE_METADATA must be set together"
        )
    if isolated_seed is None:
        runpy.run_path(str(entrypoint), run_name="__main__")
    else:
        _run_one_isolated_episode(entrypoint, int(isolated_seed), Path(metadata).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
