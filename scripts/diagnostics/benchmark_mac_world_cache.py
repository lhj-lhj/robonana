#!/usr/bin/env python3
# 中文：诊断：对比 world model 前缀缓存的数值和耗时。
# English: Diagnostic: compare world-model prefix-cache numerics and timing.
# 调用 / Invocation: 读取真实窗口与权重；写测量结果，不训练。 / Reads real windows/weights and writes measurements; no training.
# 导航 / Guide: scripts/README.md (diagnostics)
"""Paired 20-step world-cache benchmark on real training windows.

Reads a checkpoint and saved stage-1 data config, never trains or writes model
state. Report latency as contended when other jobs share the selected GPU.
The only supported execution path uses FP32 weights and computation.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from robonana.data.robotwin_hdf5 import RoboTwinHDF5Dataset
from robonana.data.robotwin_lerobot import RoboTwinLeRobotDataset
from robonana.models.pretrained import load_flux2_fact_trained_checkpoint
from robonana.sampling import prefill_mac_condition, sample_mac_world
from robonana.training.posttraining import fp32_compute_context


def restore(value):
    if isinstance(value, dict):
        return {key: restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(restore(x) for x in value[1:]) if value and value[0] == "__tuple__" else [restore(x) for x in value]
    return value


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--windows-per-pool", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Repeat each window to exercise the requested compute batch shape")
    args = parser.parse_args()
    if args.repeats < 1 or args.windows_per_pool < 1 or args.batch_size < 1:
        parser.error("repeats/windows/batch must be positive")
    config = restore(json.loads(Path(args.data_config).read_text()))
    gh, gw = config["train"]["latent_grid_height"], config["train"]["latent_grid_width"]
    torch.manual_seed(20260907)
    model, _ = load_flux2_fact_trained_checkpoint(
        args.checkpoint, config_path=args.model_config, action_dim=14, state_dim=14,
        expert_hidden_dim=1024, device="cuda:0", dtype=torch.float32)
    model.eval().requires_grad_(False)
    print(json.dumps(dict(device=torch.cuda.get_device_name(), precision="fp32",
                          weights="fp32", steps=20, batch=args.batch_size, repeats=args.repeats)), flush=True)
    classes = {cls.__name__: cls for cls in (RoboTwinHDF5Dataset, RoboTwinLeRobotDataset)}
    weights = config["dataloaders"]["train"]["sampler"]["pool_weights"]
    for pool in config["dataloaders"]["train"]["data_or_config"]:
        if weights.get(pool["pool_name"], 0) == 0:
            continue
        dataset = classes[pool["_class_name"]].load(pool)
        dataset.open()
        try:
            for record_index, record in enumerate(dataset.records[:args.windows_per_pool]):
                frame = max(0, record.length - 1 - 48)
                item = dataset[int(dataset.episode_starts[record_index]) + frame]
                def batch(key):
                    return item[key][None].to("cuda", dtype=torch.float32).repeat_interleave(args.batch_size, dim=0)
                inputs = dict(context=batch("context"), current_latents=batch("current_latents"),
                              state=batch("state")[:, None], context_mask=item["context_mask"][None].cuda().repeat_interleave(args.batch_size, dim=0))
                kwargs = dict(model=model, **inputs, clean_action=batch("behavior_action"),
                              future_noise=torch.randn_like(inputs["current_latents"]),
                              future_state_noise=torch.randn_like(inputs["state"]),
                              schedule=torch.linspace(1, 0, 21, device="cuda"), grid_height=gh, grid_width=gw)
                results, timings, peaks = {}, {False: [], True: []}, {False: [], True: []}
                with fp32_compute_context(model):
                    # C exists already after stage-2 action selection. Include
                    # G/R/U prefill in cached timing, not this shared C prefill.
                    cache = prefill_mac_condition(model=model, **inputs, grid_height=gh, grid_width=gw)
                    for mode in (False, True):
                        sample_mac_world(**kwargs, use_cache=mode, condition_cache=cache if mode else None)
                    for repeat in range(args.repeats):
                        for mode in ((False, True) if repeat % 2 == 0 else (True, False)):
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            start = time.perf_counter()
                            output = sample_mac_world(**kwargs, use_cache=mode, condition_cache=cache if mode else None)
                            torch.cuda.synchronize()
                            timings[mode].append(time.perf_counter() - start)
                            peaks[mode].append(torch.cuda.max_memory_allocated() / 2**30)
                            results[mode] = {key: getattr(output, key).float().cpu() for key in
                                             ("future", "future_state", "reward_logits", "success_logit")}
                    errors = {}
                    for key, actual in results[True].items():
                        expected = results[False][key]
                        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                        delta = actual - expected
                        errors[key] = dict(max_abs=delta.abs().max().item(), rmse=delta.square().mean().sqrt().item(),
                                           reference_rms=expected.square().mean().sqrt().item())
                    p0, p1 = (results[mode]["success_logit"][0].sigmoid().item() for mode in (False, True))
                    print(json.dumps(dict(pool=pool["pool_name"], episode=record_index, frame=frame,
                        label_success=bool(item["success"].item()), full_seconds=statistics.median(timings[False]),
                        cached_seconds=statistics.median(timings[True]),
                        speedup=statistics.median(timings[False]) / statistics.median(timings[True]),
                        full_peak_gib=max(peaks[False]), cached_peak_gib=max(peaks[True]), errors=errors,
                        full_success_probability=p0, cached_success_probability=p1,
                        terminal_agrees=(p0 >= 0.5) == (p1 >= 0.5))), flush=True)
        finally:
            dataset.close()


if __name__ == "__main__":
    main()
