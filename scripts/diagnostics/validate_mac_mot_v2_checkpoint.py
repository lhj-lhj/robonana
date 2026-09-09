#!/usr/bin/env python3
# 中文：诊断：审查 MAC checkpoint 的结构及 critic 参数。
# English: Diagnostic: inspect MAC checkpoint structure and critic parameters.
# 调用 / Invocation: 可选 GPU smoke forward；不训练、不改权重。 / Optional GPU smoke forward; never trains or modifies weights.
# 导航 / Guide: scripts/README.md (diagnostics)
"""Audit a complete mac_mot_v2 checkpoint and its critic surface."""

from __future__ import annotations

import argparse
import json

import torch

from robonana.models.pretrained import configure_trainable_parameters, load_flux2_fact_trained_checkpoint
from robonana.models.position_ids import image_position_ids, text_position_ids
from robonana.training.posttraining import ValueExpertEMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="complete mac_mot_v2 checkpoint")
    parser.add_argument("--model-config", required=True, help="saved MAC config.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expert-hidden-dim", type=int, default=1024)
    parser.add_argument("--smoke-forward", action="store_true")
    args = parser.parse_args()
    model, report = load_flux2_fact_trained_checkpoint(
        args.checkpoint,
        action_dim=14,
        state_dim=14,
        reward_dim=48,
        chunk_horizon=48,
        expert_hidden_dim=args.expert_hidden_dim,
        device=args.device,
        dtype=torch.bfloat16,
    )
    trainable = configure_trainable_parameters(model, "critic")
    old_modules = [
        name for name in ("horizon_embed", "q_in", "q_out", "value_token", "q_token")
        if hasattr(model, name)
    ]
    if old_modules:
        raise RuntimeError(f"obsolete modules survived migration: {old_modules}")
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith(("value_expert.", "q_expert."))
    ):
        raise RuntimeError("critic phase left a FLUX actor/world parameter trainable")
    payload = {
        "architecture_version": model.architecture_version,
        "chunk_horizon": model.chunk_horizon,
        "expert_hidden_dim": model.expert_hidden_dim,
        "loaded_parameter_tensors": len(report.loaded_parameter_names),
        "skipped_checkpoint_tensors": len(report.skipped_checkpoint_parameters),
        "critic_trainable_tensors": len(trainable),
        "value_expert_parameters": sum(p.numel() for p in model.value_expert.parameters()),
        "q_expert_parameters": sum(p.numel() for p in model.q_expert.parameters()),
        "flux_trainable_parameters": sum(
            p.numel()
            for name, p in model.named_parameters()
            if not name.startswith(("value_expert.", "q_expert.")) and p.requires_grad
        ),
        "has_target_q": False,
        "has_ema_flux": False,
    }
    if args.smoke_forward:
        device = torch.device(args.device)
        batch = 1
        context = torch.randn(
            batch,
            2,
            model.txt_in.in_features,
            device=device,
            dtype=torch.bfloat16,
        )
        current = torch.randn(
            batch, 12 * 24, model.in_channels, device=device, dtype=torch.bfloat16
        )
        state = torch.randn(batch, 1, model.state_dim, device=device, dtype=torch.bfloat16)
        action = torch.randn(
            batch, model.chunk_horizon, model.action_dim, device=device, dtype=torch.bfloat16
        )
        context_ids = text_position_ids(batch, context.shape[1], device)
        current_ids = image_position_ids(
            batch,
            grid_height=12,
            grid_width=24,
            time_coord=torch.zeros(batch, device=device, dtype=torch.long),
            device=device,
        )
        model.train()
        value = model.predict_value(
            context=context,
            context_ids=context_ids,
            current_latents=current,
            current_ids=current_ids,
            state=state,
            context_mask=torch.ones(batch, context.shape[1], device=device, dtype=torch.bool),
        )
        q = model.predict_q(
            context=context,
            context_ids=context_ids,
            current_latents=current,
            current_ids=current_ids,
            state=state,
            clean_action=action,
            context_mask=torch.ones(batch, context.shape[1], device=device, dtype=torch.bool),
        )
        # Target Value stores FP32 weights; predict_value uses BF16 autocast.
        target_value = ValueExpertEMA(
            model.value_expert,
            device=device,
        ).model
        target_value_prediction = model.predict_value(
            context=context,
            context_ids=context_ids,
            current_latents=current,
            current_ids=current_ids,
            state=state,
            context_mask=torch.ones(batch, context.shape[1], device=device, dtype=torch.bool),
            expert=target_value,
        )
        (value.float().mean() + q.float().mean()).backward()
        flux_gradients = [
            name
            for name, parameter in model.named_parameters()
            if not name.startswith(("value_expert.", "q_expert."))
            and parameter.grad is not None
        ]
        if flux_gradients:
            raise RuntimeError(f"critic backward reached frozen FLUX: {flux_gradients[:5]}")
        payload.update(
            smoke_value=float(value.detach().float().item()),
            smoke_q=float(q.detach().float().item()),
            smoke_target_value=float(target_value_prediction.detach().float().item()),
            value_expert_has_grad=any(
                parameter.grad is not None for parameter in model.value_expert.parameters()
            ),
            q_expert_has_grad=any(
                parameter.grad is not None for parameter in model.q_expert.parameters()
            ),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
