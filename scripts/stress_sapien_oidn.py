#!/usr/bin/env python3
"""Exercise SAPIEN's Vulkan/CUDA OIDN hand-off without loading a policy model."""

from __future__ import annotations

import argparse
import time

import numpy as np

from robonana.sim import configure_sapien_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=192)
    args = parser.parse_args()
    if args.frames < 1 or args.width < 1 or args.height < 1:
        parser.error("frames, width, and height must be positive")
    return args


def main() -> int:
    args = parse_args()
    configure_sapien_runtime()

    import sapien

    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser("oidn")

    scene = sapien.Scene()
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [1, 1, 1], shadow=True)

    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.1, 0.1, 0.1])
    builder.add_box_visual(half_size=[0.1, 0.1, 0.1], material=[0.8, 0.1, 0.1])
    actor = builder.build(name="oidn_stress_box")
    actor.set_pose(sapien.Pose([0, 0, 0.1]))

    camera = scene.add_camera("oidn_stress_camera", args.width, args.height, 1.0, 0.01, 10)
    camera.set_pose(sapien.Pose([1.1, 0, 0.65], [0.7071068, 0, 0.7071068, 0]))

    started = time.monotonic()
    checksum = 0.0
    for index in range(args.frames):
        actor.set_pose(sapien.Pose([0, 0, 0.1 + 0.025 * np.sin(index * 0.03)]))
        scene.step()
        scene.update_render()
        camera.take_picture()
        color = camera.get_picture("Color")
        checksum += float(color[0, 0, 0])
        if (index + 1) % 50 == 0 or index + 1 == args.frames:
            print(f"frames={index + 1}/{args.frames}", flush=True)

    duration = time.monotonic() - started
    print(
        f"completed frames={args.frames} duration_seconds={duration:.3f} "
        f"fps={args.frames / duration:.3f} checksum={checksum:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
