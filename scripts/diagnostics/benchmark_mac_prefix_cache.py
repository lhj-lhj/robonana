#!/usr/bin/env python3
# 中文：诊断：对比 Q 候选前缀缓存与参考实现的数值及性能。
# English: Diagnostic: compare Q-prefix caching against the reference implementation.
# 调用 / Invocation: 显式 GPU 测试；不改模型权重。 / Explicit GPU test; never changes model weights.
# 导航 / Guide: scripts/README.md (diagnostics)
"""Measure cached rejection against the full asymmetric-forward reference.

The reference intentionally repeats C and evaluates unused Value, matching the
old execution cost, but uses the corrected Q mask for a fair semantic check.
Synthetic observations measure compute, not environment policy success.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from robonana.models.position_ids import image_position_ids, text_position_ids
from robonana.models.pretrained import load_flux2_fact_trained_checkpoint
from robonana.sampling import (
    QRejectionSample, flow_euler_schedule, sample_action_flow, sample_q_rejection,
)


def full_rejection(model, inputs, noise, schedule):
    """Uncached oracle built from public full-FLUX forwards, no new block math."""
    batch, count, horizon, dim = noise.shape
    flat_batch = batch * count
    flat = {key: value.repeat_interleave(count, dim=0) for key, value in inputs.items()}
    device = noise.device
    common = dict(
        **flat, context_ids=text_position_ids(flat_batch, flat["context"].shape[1], device),
        current_ids=image_position_ids(flat_batch, grid_height=12, grid_width=24,
                                      time_coord=torch.zeros(flat_batch, device=device, dtype=torch.long), device=device),
    )
    empty_action = noise.new_empty(flat_batch, 0, dim)
    empty_state = noise.new_empty(flat_batch, 0, model.state_dim)
    empty_image = noise.new_empty(flat_batch, 0, model.in_channels)
    empty_scalar = noise.new_empty(flat_batch, 0, 1)
    zeros = torch.zeros(flat_batch, device=device)

    def velocity(action, sigma):
        return model(
            **common, noisy_pred_action=action, gt_action_cond=empty_action,
            noisy_future_latents=empty_image, future_ids=torch.empty(flat_batch, 0, 4, device=device, dtype=torch.long),
            noisy_future_state=empty_state, noisy_reward=empty_scalar, noisy_q=empty_scalar,
            chunk_horizon=torch.full((flat_batch,), 48, device=device),
            action_timestep=sigma.expand(flat_batch), wm_timestep=zeros,
        ).action

    actions = sample_action_flow(action_noise=noise.reshape(flat_batch, horizon, dim),
                                schedule=schedule, predict_action=velocity)
    # This is deliberately the uncached BOTH path, not predict_q(), which is
    # optimized in production. Keep the reference independent of that path.
    model.predict_value(**common)
    cache = model.prefill_critic_cache(**common, clean_action=actions)
    pe = model._expert_query_pe(batch=flat_batch, device=device, dtype=torch.long, segment_id=11)
    scores = model.q_expert(cache, query_pe=pe).reshape(batch, count)
    candidates = actions.reshape(batch, count, horizon, dim)
    best = scores.argmax(dim=1)
    return QRejectionSample(candidates[torch.arange(batch, device=device), best], candidates, scores, best)


def measure(fn, repeats):
    warmup = fn()
    del warmup
    torch.cuda.synchronize()
    times, peaks = [], []
    result = None
    for _ in range(repeats):
        del result
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        peaks.append(torch.cuda.max_memory_allocated() / 2**30)
    return result, dict(seconds_median=statistics.median(times), peak_allocated_gib=max(peaks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(42)
    model, _ = load_flux2_fact_trained_checkpoint(
        args.checkpoint, config_path=args.model_config, action_dim=14, state_dim=14,
        expert_hidden_dim=1024, device="cuda:0", dtype=torch.float32,
    )
    model.set_training_phase("critic")
    model.eval()
    inputs = dict(
        context=torch.randn(1, 512, model.txt_in.in_features, device="cuda", dtype=torch.float32),
        current_latents=torch.randn(1, 288, model.in_channels, device="cuda", dtype=torch.float32),
        state=torch.randn(1, 1, 14, device="cuda", dtype=torch.float32),
        context_mask=torch.ones(1, 512, device="cuda", dtype=torch.bool),
    )
    schedule = flow_euler_schedule(args.sampling_steps, flow_shift=1.0, device="cuda")
    print(json.dumps(dict(device=torch.cuda.get_device_name(), sampling_steps=args.sampling_steps,
                          repeats=args.repeats, batch=1, language_tokens=512, image_tokens=288)), flush=True)
    with torch.inference_mode():
        for count in args.counts:
            noise = torch.randn(1, count, 48, 14, device="cuda", dtype=torch.float32)
            full, baseline = measure(lambda: full_rejection(model, inputs, noise, schedule), args.repeats)
            full_actions, full_q = full.candidates.cpu(), full.candidate_q.cpu()
            full_best = full.best_index.item()
            del full
            print(json.dumps(dict(count=count, mode="full_reference", **baseline)), flush=True)
            for group in args.group_sizes:
                cached, metrics = measure(lambda: sample_q_rejection(
                    model=model, **inputs, action_noise=noise, schedule=schedule,
                    candidate_count=count, candidate_batch_size=group, grid_height=12, grid_width=24,
                ), args.repeats)
                print(json.dumps(dict(
                    count=count, mode="cached", group_size=group, **metrics,
                    speedup=baseline["seconds_median"] / metrics["seconds_median"],
                    action_max_abs_error=(cached.candidates.cpu().float() - full_actions.float()).abs().max().item(),
                    q_max_abs_error=(cached.candidate_q.cpu().float() - full_q.float()).abs().max().item(),
                    selected_index=cached.best_index.item(), reference_index=full_best,
                )), flush=True)
                assert torch.isfinite(cached.candidates).all() and torch.isfinite(cached.candidate_q).all()
                del cached


if __name__ == "__main__":
    main()
