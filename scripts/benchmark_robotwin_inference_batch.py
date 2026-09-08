#!/usr/bin/env python3
"""Compare existing policy batching on identical recorded observations/noise.

No new model/sampler math: use BatchedRoboNanaRobotWinPolicy.inference_batch.
The three settings isolate candidate grouping from cross-environment batching.
JPEG observations are decoded once; this is a compute/numerical probe, not an SR
evaluation or a replay of the source episode's actions. Probe seeds are explicit.
"""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import h5py
import numpy as np
from PIL import Image
import torch

from robonana.inference.batched_policy import BatchedRoboNanaRobotWinPolicy
from world_action_model import apply_runtime_compat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--flux-checkpoint-dir", required=True)
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--source-episodes", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    apply_runtime_compat()
    observations, provenance = [], []
    cameras = {"cam_high": "head_camera", "cam_left_wrist": "left_camera",
               "cam_right_wrist": "right_camera"}
    for path in args.source_episodes:
        with h5py.File(path, "r") as handle:
            # First and final nonterminal planning boundaries, including failure tails.
            for frame in (0, ((len(handle["joint_action/vector"]) - 2) // 48) * 48):
                seed = 12345 + len(observations)
                obs = {"observation.state": handle["joint_action/vector"][frame],
                       "instruction": str(handle.attrs["instruction"]), "sampling_seed": seed}
                for key, camera in cameras.items():
                    encoded = handle[f"observation/{camera}/rgb"][frame]
                    with Image.open(BytesIO(bytes(encoded))) as image:
                        obs[f"observation.images.{key}"] = np.array(image.convert("RGB"))
                observations.append(obs)
                provenance.append({"source": str(path), "frame": frame, "sampling_seed": seed})
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key != "source_episodes"}
    config.update(observations=provenance, commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip())
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    policy = BatchedRoboNanaRobotWinPolicy(
        checkpoint=args.checkpoint, model_config=args.model_config,
        flux_checkpoint_dir=args.flux_checkpoint_dir, stats_path=args.stats_path,
        model_device="cuda:0", vae_device="cuda:0", text_encoder_device="cuda:0",
        dtype=torch.float32, action_chunk=48, horizon=48, num_inference_steps=20,
        inference_mode="action_q_rejection", rejection_candidate_count=32, q_return_scale=1000.)

    def infer(request_batch):
        candidates, scores, indices = [], [], []
        for start in range(0, len(observations), request_batch):
            policy.inference_batch(observations[start:start + request_batch])
            result = policy._last_batch_rejection
            # CPU copies are identical instrumentation in all configurations.
            candidates.append(result.candidates.cpu())
            scores.append(result.candidate_q.cpu())
            indices.append(result.best_index.cpu())
        return torch.cat(candidates), torch.cat(scores), torch.cat(indices)

    reference = None
    reports = []
    with torch.inference_mode():
        for request_batch, candidate_batch in ((1, 16), (1, 32), (2, 32)):
            os.environ["ROBONANA_REJECTION_CANDIDATE_BATCH_SIZE"] = str(candidate_batch)
            torch.cuda.empty_cache()
            infer(request_batch)  # Warm encoders, language cache and model kernels.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            seconds = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = infer(request_batch)
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - started)
            if reference is None:
                reference = result
            row = {"request_batch": request_batch, "candidate_batch": candidate_batch,
                   "observations": len(observations), "seconds": seconds,
                   "median_seconds": statistics.median(seconds),
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                   "all_finite": all(bool(torch.isfinite(t).all()) for t in result),
                   "candidate_action_max_abs_error": (result[0] - reference[0]).abs().max().item(),
                   "q_max_abs_error_return_units": ((result[1] - reference[1]) * 1000).abs().max().item(),
                   "selected_indices": result[2].tolist(),
                   "reference_indices": reference[2].tolist(),
                   "all_indices_equal": torch.equal(result[2], reference[2])}
            reports.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
            if not row["all_finite"]:
                raise RuntimeError("nonfinite batched output; do not start collection")
    if not all(row["all_indices_equal"] for row in reports):
        raise RuntimeError("argmax changed on fixed inputs; review errors before collection")


if __name__ == "__main__":
    main()
