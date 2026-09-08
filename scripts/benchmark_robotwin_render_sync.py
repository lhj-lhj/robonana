#!/usr/bin/env python3
"""Replay saved actions through unmodified RoboTwin setup/physics.

Run each baseline/repeat/deferred variant in a fresh process on the SAME GPU.
Uses upstream eval_policy.main for configuration (no copied simulator). The
policy/server and expert precheck are intentionally excluded from timed replay.
Raw RGB comparisons include every control step, including the terminal image.
This script does not enable the optimization in production collection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time

import h5py
import numpy as np

from robonana.sim import configure_sapien_runtime
from robonana.sim.render_sync_probe import defer_action_render_sync
from robotwin_eval_bootstrap import _install_static_camera_filter

CAMERAS = ("head_camera", "left_camera", "right_camera")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--defer", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    opts = parser.parse_args()
    opts.output = opts.output.resolve()
    opts.output.mkdir(parents=True, exist_ok=False)
    with h5py.File(opts.episode.resolve(), "r") as source:
        valid = np.asarray(source["transition_valid"], dtype=bool)
        actions = np.asarray(source["policy_action/vector"])[valid]
        recorded_states = np.asarray(source["joint_action/vector"])
        seed = int(source.attrs["seed"])
        task_name = str(source.attrs["task_name"])
        task_config = str(source.attrs["task_config"])
        instruction = str(source.attrs["instruction"])
    if opts.max_steps:
        actions = actions[:opts.max_steps]
    os.chdir(opts.robotwin.resolve())
    sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "script")]
    configure_sapien_runtime()
    # Same CuRobo/Warp compatibility as the production RoboNana client.
    import warp
    if not hasattr(warp, "torch"):
        from warp._src import torch as warp_torch
        warp.torch = warp_torch
    namespace = runpy.run_path("script/eval_policy.py", run_name="_render_probe")
    _install_static_camera_filter()

    def replay(_task_name, task, args, _model, _seed, **kwargs):
        args.update(eval_mode=True, render_freq=0, eval_video_log=False)
        args.pop("eval_video_save_dir", None)
        random.seed(seed)
        setup_start = time.perf_counter()
        task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        task.set_instruction(instruction=instruction)
        setup_seconds = time.perf_counter() - setup_start
        states, successes, timings = [], [], []
        rgb_store = None
        reference_rgb = None
        if opts.reference:
            reference_rgb = np.load(opts.reference / "rgb.npy", mmap_mode="r")
        differences = []
        try:
            with defer_action_render_sync(task, enabled=opts.defer) as counts:
                start = time.perf_counter()
                obs = task.get_obs()
                initial_obs_seconds = time.perf_counter() - start
                for step in range(len(actions) + 1):
                    rgb = np.stack([obs["observation"][name]["rgb"] for name in CAMERAS])
                    if rgb_store is None:
                        rgb_store = np.lib.format.open_memmap(
                            opts.output / "rgb.npy", mode="w+", dtype=rgb.dtype,
                            shape=(len(actions) + 1, *rgb.shape),
                        )
                    rgb_store[step] = rgb
                    if reference_rgb is not None and step < len(reference_rgb):
                        diff = np.abs(rgb.astype(np.int16) - reference_rgb[step].astype(np.int16))
                        differences.append({"mae": float(diff.mean()), "max": int(diff.max()),
                                            "different_fraction": float(np.mean(diff != 0))})
                    states.append(np.asarray(obs["joint_action"]["vector"]).copy())
                    successes.append(bool(task.eval_success))
                    if task.eval_success or step == len(actions):
                        break
                    # Only simulator work is timed; dump/compare raw RGB outside.
                    start = time.perf_counter()
                    task.take_action(actions[step].copy())
                    action_end = time.perf_counter()
                    obs = task.now_obs if task.eval_success else task.get_obs()
                    end = time.perf_counter()
                    timings.append((action_end - start, end - action_end))
                    if step % 48 == 0:
                        print(f"step={step} action_s={timings[-1][0]:.4f} obs_s={timings[-1][1]:.4f}", flush=True)
                result = {
                    "seed": seed, "defer": opts.defer, "steps": len(timings),
                    "success": bool(task.eval_success), "setup_seconds": setup_seconds,
                    "initial_obs_seconds": initial_obs_seconds,
                    "action_seconds": float(np.sum(timings, axis=0)[0]),
                    "observation_seconds": float(np.sum(timings, axis=0)[1]),
                    "render_counts": dict(counts), "rgb_difference_by_frame": differences,
                    "recorded_state_max_abs": float(np.max(np.abs(np.asarray(states) - recorded_states[:len(states)]))),
                }
                if opts.reference:
                    reference_states = np.load(opts.reference / "states.npy")
                    result["reference_state_shape_equal"] = list(reference_states.shape) == list(np.asarray(states).shape)
                    if result["reference_state_shape_equal"]:
                        result["reference_state_max_abs"] = float(np.max(np.abs(reference_states - states)))
                    result["reference_success_flags_equal"] = np.array_equal(np.load(opts.reference / "success.npy"), successes)
                np.save(opts.output / "states.npy", states)
                np.save(opts.output / "success.npy", successes)
                np.save(opts.output / "timings.npy", timings)
                rgb_store.flush()
                (opts.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(json.dumps({k: v for k, v in result.items() if k != "rgb_difference_by_frame"}), flush=True)
        finally:
            task.close_env()
        return seed + 1, int(task.eval_success)

    # Reuse upstream configuration resolution, replace only the policy loop.
    main_fn = namespace["main"]
    main_fn.__globals__["eval_function_decorator"] = lambda *args: lambda _args: None
    main_fn.__globals__["eval_policy"] = replay
    main_fn({"task_name": task_name, "task_config": task_config,
             "ckpt_setting": "render_sync_probe", "policy_name": "render_sync_probe",
             "instruction_type": "unseen", "seed": 0, "test_num": 1})


if __name__ == "__main__":
    main()
