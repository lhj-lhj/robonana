#!/usr/bin/env python3
# 中文：诊断：对比 world model 前缀缓存的数值和耗时。
# English: Diagnostic: compare world-model prefix-cache numerics and timing.
# 调用 / Invocation: 读取真实窗口与权重；写测量结果，不训练。 / Reads real windows/weights and writes measurements; no training.
# 导航 / Guide: scripts/README.md (diagnostics)
"""Paired 20-step world-cache benchmark on real training windows.

Reads a checkpoint and saved stage-1 data config, never trains or writes model
state. Report latency as contended when other jobs share the selected GPU.
Match FACT's BF16 model execution.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch

import torch

from robonana.data.robotwin_hdf5 import RoboTwinHDF5Dataset
from robonana.data.robotwin_lerobot import RoboTwinLeRobotDataset
from robonana.models.pretrained import load_flux2_fact_trained_checkpoint
from robonana.sampling import prefill_mac_condition, sample_mac_world


def restore(value):
    if isinstance(value, dict):
        return {key: restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(restore(x) for x in value[1:]) if value and value[0] == "__tuple__" else [restore(x) for x in value]
    return value


def exclusive_times(times):
    """Remove inclusive parents so percentages never double-count GPU work."""
    result = dict(times)
    result["prefix_selection_overhead"] = result.pop("rejection") - result["action"] - result["q_score"]
    result["bootstrap_other"] = result.pop("rollout") - (
        result["prefix_selection_overhead"] + result["action"] + result["q_score"] + result["world"])
    return result


def stage2_breakdown(args, config):
    """中文：复用生产采样；共享 GPU 的 BS1 诊断，不代表八卡吞吐。

    English: Time the production sampler with synchronized, non-overlapping
    wall-clock scopes. Updates affect this disposable model only. Local AdamW
    is a timing proxy, NOT DeepSpeed/ZeRO or distributed-training equivalence.
    Dataset loading, checkpoint loading and warmup are excluded.
    """
    import robonana.sampling as sampling
    from robonana.training.losses import deterministic_return_loss
    from robonana.training.posttraining import ValueExpertEMA, evaluating
    from safetensors.torch import load_file

    model, _ = load_flux2_fact_trained_checkpoint(
        args.checkpoint, config_path=args.model_config, action_dim=14, state_dim=14,
        expert_hidden_dim=1024, device="cuda:0", dtype=torch.bfloat16)
    model.set_training_phase("critic")
    model.train()
    post = config["train"]["posttrain"]
    ema = ValueExpertEMA(model.value_expert, **{
        k: v for k, v in post["ema"].items() if k != "target"})
    if args.target_value:
        ema.model.load_state_dict(load_file(args.target_value), strict=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    classes = {cls.__name__: cls for cls in (RoboTwinHDF5Dataset, RoboTwinLeRobotDataset)}
    pool = next(p for p in config["dataloaders"]["train"]["data_or_config"]
                if config["dataloaders"]["train"]["sampler"]["pool_weights"].get(p["pool_name"], 0) > 0)
    dataset = classes[pool["_class_name"]].load(pool)
    dataset.open()
    try:
        item = dataset[0]
    finally:
        dataset.close()
    def batch(key):
        return item[key][None].to("cuda", dtype=torch.bfloat16).repeat_interleave(args.batch_size, 0)
    inputs = dict(context=batch("context"), current_latents=batch("current_latents"),
                  state=batch("state")[:, None],
                  context_mask=item["context_mask"][None].cuda().repeat_interleave(args.batch_size, 0),
                  grid_height=config["train"]["latent_grid_height"],
                  grid_width=config["train"]["latent_grid_width"])
    imag = post["imagination"]
    schedule = sampling.flow_euler_schedule(imag["sampling_steps"],
                flow_shift=imag["flow_shift"], device="cuda")
    print(json.dumps(dict(mode="stage2_breakdown", batch=args.batch_size,
        candidates=imag["candidate_count"], sampling_steps=imag["sampling_steps"],
        shared_gpu=True, distributed=False, optimizer="local_BF16_AdamW_proxy",
        checkpoint=args.checkpoint, data_pool=pool["pool_name"])), flush=True)
    rows = []
    for step in range(args.repeats + 2):
        times = {}
        @contextmanager
        def measure(name):
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                yield
            finally:
                torch.cuda.synchronize()
                times[name] = times.get(name, 0.0) + time.perf_counter() - start
        def timed(name, fn):
            def wrapped(*a, **kw):
                with measure(name):
                    return fn(*a, **kw)
            return wrapped
        # 中文：仅在诊断进程包装现有函数，不复制采样/target公式。
        # English: Wrap production calls only; subtract nested inclusive scopes.
        with patch.object(sampling, "sample_q_rejection", timed("rejection", sampling.sample_q_rejection)), \
             patch.object(sampling, "sample_action_flow", timed("action", sampling.sample_action_flow)), \
             patch.object(model, "score_q_candidates", timed("q_score", model.score_q_candidates)), \
             patch.object(sampling, "sample_mac_world", timed("world", sampling.sample_mac_world)):
            with measure("rollout"), evaluating(model), torch.autocast("cuda", dtype=torch.bfloat16):
                rollout = sampling.generate_mac_imaginary_rollout_h1(
                    online_model=model, target_value_expert=ema.model, **inputs,
                    candidate_count=imag["candidate_count"],
                    action_noise=torch.randn(args.batch_size, imag["candidate_count"], 48, 14, device="cuda", dtype=torch.bfloat16),
                    future_noise=torch.randn_like(inputs["current_latents"]),
                    future_state_noise=torch.randn_like(inputs["state"]), schedule=schedule,
                    **{k: post[k] for k in ("discount", "reward_non_goal", "reward_goal", "return_scale")})
        times = exclusive_times(times)
        with measure("critic_forward_loss"), torch.autocast("cuda", dtype=torch.bfloat16):
            v, q = sampling.evaluate_mac_critics(model=model, **inputs,
                clean_action=rollout.selected_action, condition_cache=rollout.condition_cache)
            loss = sum(deterministic_return_loss(p, t, return_scale=post["return_scale"])
                       for p, t in ((v, rollout.value_target_return), (q, rollout.q_target_return)))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite diagnostic loss")
        with measure("backward"):
            loss.backward()
        with measure("optimizer_proxy"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with measure("ema"):
            ema.update(model.value_expert, optimizer_step=step + 1, optimizer_step_succeeded=True)
        del rollout, v, q, loss
        if step >= 2:
            rows.append(times)
        print(json.dumps(dict(step=step, warmup=step < 2, seconds=times)), flush=True)
    means = {k: statistics.mean(row[k] for row in rows) for k in rows[0]}
    total = sum(means.values())
    print(json.dumps(dict(summary=True, mean_seconds=means, total_seconds=total,
        percent={k: 100 * v / total for k, v in means.items()},
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)), flush=True)


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
    parser.add_argument("--stage2-breakdown", action="store_true",
                        help="Disposable Q/V update timing; no weights saved; no distributed communication")
    parser.add_argument("--target-value", help="Optional checkpoint target_value_expert.safetensors")
    args = parser.parse_args()
    if args.repeats < 1 or args.windows_per_pool < 1 or args.batch_size < 1:
        parser.error("repeats/windows/batch must be positive")
    config = restore(json.loads(Path(args.data_config).read_text()))
    if args.stage2_breakdown:
        with torch.inference_mode(False), torch.enable_grad():
            stage2_breakdown(args, config)
        return
    gh, gw = config["train"]["latent_grid_height"], config["train"]["latent_grid_width"]
    torch.manual_seed(20260907)
    model, _ = load_flux2_fact_trained_checkpoint(
        args.checkpoint, config_path=args.model_config, action_dim=14, state_dim=14,
        expert_hidden_dim=1024, device="cuda:0", dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    print(json.dumps(dict(device=torch.cuda.get_device_name(), precision="bf16",
                          weights="bf16", steps=20, batch=args.batch_size, repeats=args.repeats)), flush=True)
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
                    return item[key][None].to("cuda", dtype=torch.bfloat16).repeat_interleave(args.batch_size, dim=0)
                inputs = dict(context=batch("context"), current_latents=batch("current_latents"),
                              state=batch("state")[:, None], context_mask=item["context_mask"][None].cuda().repeat_interleave(args.batch_size, dim=0))
                kwargs = dict(model=model, **inputs, clean_action=batch("behavior_action"),
                              future_noise=torch.randn_like(inputs["current_latents"]),
                              future_state_noise=torch.randn_like(inputs["state"]),
                              schedule=torch.linspace(1, 0, 21, device="cuda"), grid_height=gh, grid_width=gw)
                results, timings, peaks = {}, {False: [], True: []}, {False: [], True: []}
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
