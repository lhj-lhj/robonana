"""Periodic FLUX.2 latent decoding for training-time W&B monitoring."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def unpack_flux2_tokens(tokens: Tensor, vae, *, grid_height: int = 12, grid_width: int = 24) -> Tensor:
    """Invert RoboNana's FLUX patchify + BatchNorm cache representation."""

    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [B, N, C], got {tuple(tokens.shape)}")
    batch, count, packed_channels = tokens.shape
    if count != grid_height * grid_width:
        raise ValueError(f"token count {count} does not match grid {grid_height}x{grid_width}")
    if packed_channels % 4:
        raise ValueError(f"packed channels must be divisible by four, got {packed_channels}")

    packed = tokens.transpose(1, 2).reshape(batch, packed_channels, grid_height, grid_width)
    eps = float(getattr(getattr(vae, "config", None), "batch_norm_eps", 1e-4))
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=packed.device, dtype=packed.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1).to(device=packed.device, dtype=packed.dtype) + eps)
    packed = packed * std + mean

    channels = packed_channels // 4
    raw = packed.reshape(batch, channels, 2, 2, grid_height, grid_width)
    return raw.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels, grid_height * 2, grid_width * 2)


@torch.no_grad()
def decode_flux2_tokens(vae, tokens: Tensor, *, grid_height: int = 12, grid_width: int = 24) -> Tensor:
    raw = unpack_flux2_tokens(tokens, vae, grid_height=grid_height, grid_width=grid_width)
    decoded = vae.decode(raw.to(dtype=next(vae.parameters()).dtype), return_dict=False)[0]
    return decoded.float().add(1.0).div(2.0).clamp(0.0, 1.0)


def should_log_pixel_eval(step: int, interval: int) -> bool:
    return interval > 0 and step > 0 and step % interval == 0


def log_pixel_eval(
    *,
    accelerator,
    step: int,
    current: Tensor,
    targets: Tensor,
    predictions: Tensor,
    gt_action_predictions: Tensor | None = None,
    horizons: Tensor,
    num_inference_steps: int,
) -> None:
    """Upload fixed-horizon samples gathered from every distributed rank."""

    if not accelerator.is_main_process:
        return
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Pixel eval logging requires wandb") from error

    current_images = current.detach().cpu()
    target_images = targets.detach().cpu()
    prediction_images = predictions.detach().cpu()
    gt_action_prediction_images = (
        None if gt_action_predictions is None else gt_action_predictions.detach().cpu()
    )
    horizon_tensor = horizons.detach().cpu()

    if target_images.ndim != 5 or prediction_images.shape != target_images.shape:
        raise ValueError("distributed targets and predictions must have shape [R, H, C, Y, X]")
    if (
        gt_action_prediction_images is not None
        and gt_action_prediction_images.shape != target_images.shape
    ):
        raise ValueError("GT-action predictions must match distributed target shape")
    rank_count, horizon_count = target_images.shape[:2]
    if current_images.shape[0] != rank_count:
        raise ValueError("current, targets, and predictions must have the same rank dimension")
    horizon_tensor = horizon_tensor.reshape(rank_count, horizon_count)
    horizon_rows = [[int(value) for value in row.tolist()] for row in horizon_tensor]

    def build_panel(prediction_set: Tensor) -> Tensor:
        rows = []
        for rank in range(rank_count):
            cells = [current_images[rank]]
            for horizon_index in range(horizon_count):
                cells.extend([target_images[rank, horizon_index], prediction_set[rank, horizon_index]])
            rows.append(torch.cat(cells, dim=-1))
        return torch.cat(rows, dim=-2)

    predicted_action_panel = build_panel(prediction_images)
    predicted_action_caption = (
        "rows: one different current frame per rank; predicted-action conditioning; columns: current | "
        + " | ".join(f"GT h={value} | pred h={value}" for value in horizon_rows[0])
    )

    shared_horizons = horizon_rows[0]
    tracker = accelerator.get_tracker("wandb", unwrap=True)
    payload = {
        "eval/fixed_horizon_grid": wandb.Image(
            predicted_action_panel,
            caption=predicted_action_caption,
        ),
        "eval/fixed_horizons": ",".join(str(value) for value in shared_horizons),
        "eval/num_ranks": int(current_images.shape[0]),
        "eval/sampling": "two_stage_flow_euler_from_pure_noise",
        "eval/gt_action_sampling": "world_flow_euler_from_pure_noise_with_gt_action",
        "eval/num_inference_steps": int(num_inference_steps),
    }
    if gt_action_prediction_images is not None:
        gt_action_caption = (
            "rows: one different current frame per rank; GT-action conditioning; columns: current | "
            + " | ".join(f"GT h={value} | pred h={value}" for value in horizon_rows[0])
        )
        payload["eval/fixed_horizon_gt_action_grid"] = wandb.Image(
            build_panel(gt_action_prediction_images),
            caption=gt_action_caption,
        )
    tracker.log(
        payload,
        step=int(step),
        commit=False,
    )
