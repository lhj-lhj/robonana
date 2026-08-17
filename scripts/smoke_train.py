#!/usr/bin/env python3
"""One-step checkpoint-backed RoboNana graph smoke test on a shared GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flux2.model import Klein4BParams

from robonana.models.pretrained import configure_trainable_parameters, load_flux2_fact_checkpoint
from robonana.training.losses import joint_flow_loss
from robonana.training.memory import GIB, memory_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-mode", choices=("adapters", "full"), default="adapters")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--action-chunk", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--headroom-gib", type=float, default=2.0)
    parser.add_argument(
        "--memory-limit-gib",
        type=float,
        default=None,
        help="Cap this process to simulate a smaller amount of available GPU memory.",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _ids(batch_size: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch_size, length, 4, device=device, dtype=torch.float32)


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("shared-GPU smoke validation is intentionally restricted to batch_size=1")
    if not torch.cuda.is_available():
        raise RuntimeError("this smoke test requires CUDA")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if args.memory_limit_gib is not None:
        if args.memory_limit_gib <= 0:
            raise ValueError("memory-limit-gib must be positive")
        limit_bytes = int(args.memory_limit_gib * GIB)
        torch.cuda.set_per_process_memory_fraction(min(1.0, limit_bytes / total_bytes), device)
        free_bytes = min(free_bytes, limit_bytes)
    checkpoint_bytes = args.checkpoint.stat().st_size
    preflight = memory_preflight(
        checkpoint_bytes=checkpoint_bytes,
        free_bytes=free_bytes,
        mode=args.train_mode,
        headroom_bytes=int(args.headroom_gib * GIB),
    )
    print(
        json.dumps(
            {
                "event": "memory_preflight",
                "device": str(device),
                "batch_size": args.batch_size,
                "train_mode": args.train_mode,
                "free_gib": round(free_bytes / GIB, 3),
                "total_gib": round(total_bytes / GIB, 3),
                "checkpoint_gib": round(checkpoint_bytes / GIB, 3),
                "required_free_gib": round(preflight.required_free_bytes / GIB, 3),
                "process_memory_limit_gib": args.memory_limit_gib,
                "can_run": preflight.can_run,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not preflight.can_run:
        print("ROBONANA_SMOKE_SKIPPED_INSUFFICIENT_FREE_MEMORY", flush=True)
        return 3

    torch.cuda.reset_peak_memory_stats(device)
    model, report = load_flux2_fact_checkpoint(
        args.checkpoint,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        device=device,
        params=Klein4BParams(),
    )
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    trainable_names = configure_trainable_parameters(model, args.train_mode)
    model.train()

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    dtype = next(model.parameters()).dtype
    batch_size = args.batch_size
    context_len = 2
    ref_len = 2
    future_image_len = 2
    future_state_len = 1
    value_len = 1
    params = Klein4BParams()

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=dtype)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            context=randn(batch_size, context_len, params.context_in_dim),
            context_ids=_ids(batch_size, context_len, device),
            current_latents=randn(batch_size, ref_len, params.in_channels),
            current_ids=_ids(batch_size, ref_len, device),
            noisy_future_latents=randn(batch_size, future_image_len, params.in_channels),
            future_ids=_ids(batch_size, future_image_len, device),
            state=randn(batch_size, 1, args.state_dim),
            noisy_pred_action=randn(batch_size, args.action_chunk, args.action_dim),
            gt_action_cond=randn(batch_size, args.action_chunk, args.action_dim),
            horizon_idx=torch.ones(batch_size, device=device, dtype=torch.long),
            noisy_future_state=randn(batch_size, future_state_len, args.state_dim),
            noisy_value=randn(batch_size, value_len, 1),
            action_timestep=torch.rand(batch_size, device=device),
            wm_timestep=torch.rand(batch_size, device=device),
            context_mask=torch.ones(batch_size, context_len, device=device, dtype=torch.bool),
        )
        losses = joint_flow_loss(
            output,
            image_target=randn(batch_size, future_image_len, params.in_channels),
            action_target=randn(batch_size, args.action_chunk, args.action_dim),
            future_state_target=randn(batch_size, future_state_len, args.state_dim),
            value_target=randn(batch_size, value_len, 1),
        )
        loss = sum(losses.values())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(
        json.dumps(
            {
                "event": "train_step_complete",
                "checkpoint_parameters": report.checkpoint_parameters,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameter_count,
                "trainable_tensors": len(trainable_names),
                "loss": float(loss.detach()),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / GIB, 3),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / GIB, 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("ROBONANA_SMOKE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
