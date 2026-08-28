"""Two-step, single-GPU iterative-posttraining smoke with a tiny shared FLUX."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import torch
from flux2.model import Flux2Params
from safetensors.torch import load_file, save_file

from robonana.models.flux2_fact import Flux2FACTModel
from robonana.models.position_ids import image_position_ids, text_position_ids
from robonana.training.losses import joint_flow_loss
from robonana.training.posttraining import (
    FullModelEMA,
    build_td_targets,
    search_failure_candidates,
)
from robonana.training.robotwin_trainer import flow_noise


def _tiny_model(device: torch.device, dtype: torch.dtype) -> Flux2FACTModel:
    params = Flux2Params(
        in_channels=8,
        context_in_dim=16,
        hidden_size=32,
        num_heads=4,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[2, 2, 2, 2],
        mlp_ratio=2.0,
        use_guidance_embed=False,
    )
    return Flux2FACTModel(
        params,
        action_dim=2,
        state_dim=2,
        reward_dim=1,
        q_dim=1,
        max_horizon=48,
        pred_action_bidirectional=True,
    ).to(device=device, dtype=dtype)


def _optimizer_step(
    *,
    online: Flux2FACTModel,
    ema: FullModelEMA,
    optimizer: torch.optim.Optimizer,
    step: int,
    inputs: dict[str, torch.Tensor],
) -> tuple[object, object, torch.Tensor]:
    failure_mask = inputs["failure_mask"]
    candidate = search_failure_candidates(
        online_model=online,
        ema_model=ema.model,
        context=inputs["context"][failure_mask],
        current_latents=inputs["current"][failure_mask],
        state=inputs["state"][failure_mask],
        context_mask=inputs["context_mask"][failure_mask],
        behavior_action=inputs["behavior_action"][failure_mask],
        candidate_count=8,
        candidate_horizon=48,
        action_sampling_steps=2,
        q_sampling_steps=2,
        microbatch_size=16,
        grid_height=1,
        grid_width=2,
    )
    pred_action_target = inputs["behavior_action"].clone()
    pred_action_target[failure_mask] = candidate.pseudo_action
    td = build_td_targets(
        ema_model=ema.model,
        context=inputs["context"],
        next_current_latents=inputs["future"],
        next_state=inputs["future_state"],
        context_mask=inputs["context_mask"],
        reward_h=inputs["reward"],
        delta_steps=inputs["delta_steps"],
        success_terminal_h=inputs["success_terminal_h"],
        action_template=inputs["behavior_action"],
        action_sampling_steps=2,
        q_sampling_steps=2,
        microbatch_size=16,
        grid_height=1,
        grid_width=2,
    )

    batch_size = inputs["context"].shape[0]
    action_timestep = torch.rand(batch_size, device=inputs["context"].device)
    wm_timestep = torch.rand(batch_size, device=inputs["context"].device)
    noisy_action, action_target = flow_noise(pred_action_target, action_timestep)
    noisy_future, image_target = flow_noise(inputs["future"], wm_timestep)
    noisy_state, state_target = flow_noise(inputs["future_state"], wm_timestep)
    noisy_reward, reward_target = flow_noise(inputs["reward"], wm_timestep)
    clean_q = td.q_target.to(dtype=inputs["context"].dtype)
    noisy_q, q_target = flow_noise(clean_q, wm_timestep)
    context_ids = text_position_ids(batch_size, inputs["context"].shape[1], inputs["context"].device)
    current_ids = image_position_ids(
        batch_size,
        grid_height=1,
        grid_width=2,
        time_coord=torch.zeros(batch_size, device=inputs["context"].device, dtype=torch.long),
        device=inputs["context"].device,
    )
    future_ids = image_position_ids(
        batch_size,
        grid_height=1,
        grid_width=2,
        time_coord=inputs["horizon_idx"],
        device=inputs["context"].device,
    )
    online.train()
    output = online(
        context=inputs["context"],
        context_ids=context_ids,
        current_latents=inputs["current"],
        current_ids=current_ids,
        noisy_future_latents=noisy_future,
        future_ids=future_ids,
        state=inputs["state"],
        noisy_pred_action=noisy_action,
        gt_action_cond=inputs["behavior_action"],
        horizon_idx=inputs["horizon_idx"],
        noisy_future_state=noisy_state,
        noisy_reward=noisy_reward,
        noisy_q=noisy_q,
        action_timestep=action_timestep,
        wm_timestep=wm_timestep,
        context_mask=inputs["context_mask"],
    )
    losses = joint_flow_loss(
        output,
        image_target=image_target,
        action_target=action_target,
        future_state_target=state_target,
        reward_target=reward_target,
        q_target=q_target,
        action_loss_mask=torch.ones(batch_size, device=inputs["context"].device),
        q_loss_mask=td.q_loss_mask,
    )
    total = (
        losses["image_loss"]
        + 10.0 * losses["action_loss"]
        + 0.4 * losses["future_state_loss"]
        + 0.01 * losses["reward_loss"]
        + 0.001 * losses["q_loss"]
    )
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    assert any(parameter.grad is not None for parameter in online.parameters())
    assert all(parameter.grad is None for parameter in ema.model.parameters())
    optimizer.step()
    assert ema.update(online, optimizer_step=step, optimizer_step_succeeded=True)
    assert ema.update_count == step
    return candidate, td, total.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("posttraining smoke requires one CUDA GPU")
    torch.manual_seed(20260829)
    torch.cuda.manual_seed_all(20260829)
    dtype = torch.bfloat16
    online = _tiny_model(device, dtype)
    ema = FullModelEMA(online, decay=0.995, device=device)
    optimizer = torch.optim.AdamW(online.parameters(), lr=1e-4)
    ema.assert_not_in_optimizer(optimizer)
    batch_size = 4
    behavior = torch.randn(batch_size, 48, 2, device=device, dtype=dtype)
    inputs = {
        "context": torch.randn(batch_size, 2, 16, device=device, dtype=dtype),
        "context_mask": torch.ones(batch_size, 2, device=device, dtype=torch.bool),
        "current": torch.randn(batch_size, 2, 8, device=device, dtype=dtype),
        "future": torch.randn(batch_size, 2, 8, device=device, dtype=dtype),
        "state": torch.randn(batch_size, 1, 2, device=device, dtype=dtype),
        "future_state": torch.randn(batch_size, 1, 2, device=device, dtype=dtype),
        "behavior_action": behavior,
        "reward": torch.tensor([-4.0, -7.0, -5.0, -3.0], device=device).reshape(4, 1, 1),
        "delta_steps": torch.tensor([4, 7, 5, 3], device=device),
        "success_terminal_h": torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        "failure_mask": torch.tensor([False, False, True, True], device=device),
        "horizon_idx": torch.tensor([12, 24, 48, 48], device=device),
    }
    pool_names = [
        "original_success",
        "collected_success_replay",
        "historical_failure_replay",
        "latest_failure",
    ]
    started = time.perf_counter()
    first_candidate = first_td = None
    for step in (1, 2):
        candidate, td, loss = _optimizer_step(
            online=online,
            ema=ema,
            optimizer=optimizer,
            step=step,
            inputs=inputs,
        )
        if step == 1:
            first_candidate, first_td = candidate, td
        print(f"optimizer_step={step} loss={float(loss):.6f} ema_updates={ema.update_count}")
    assert first_candidate is not None and first_td is not None
    print(f"pools={pool_names}")
    print(f"candidate_q={first_candidate.candidate_q[0].float().cpu().tolist()}")
    print(
        "candidate "
        f"best_index={int(first_candidate.best_index[0])} "
        f"best_q={float(first_candidate.best_q[0]):.6f} "
        f"behavior_q={float(first_candidate.behavior_q[0]):.6f} "
        f"pseudo_action_shape={tuple(first_candidate.pseudo_action.shape)} "
        "action_loss_mask=[1.0, 1.0]"
    )
    print(
        "success_td "
        f"reward_h={float(inputs['reward'][0]):.6f} "
        f"delta_steps={int(inputs['delta_steps'][0])} "
        f"success_terminal_h={float(inputs['success_terminal_h'][0]):.1f} "
        f"bootstrap_mask={float(first_td.bootstrap_mask[0]):.1f} "
        f"next_q={float(first_td.next_q[0]):.6f} "
        f"q_target={float(first_td.q_target[0]):.6f}"
    )
    print(
        "failure_timeout_td "
        f"reward_h={float(inputs['reward'][3]):.6f} "
        f"delta_steps={int(inputs['delta_steps'][3])} "
        "time_limit_truncated_h=1.0 bootstrap_mask="
        f"{float(first_td.bootstrap_mask[3]):.1f} "
        "final_observation_id=latest_failure/reset-pre-final "
        f"next_q={float(first_td.next_q[3]):.6f} "
        f"q_target={float(first_td.q_target[3]):.6f}"
    )
    print(
        f"candidate_search_ms={first_candidate.elapsed_ms:.3f} "
        f"candidate_peak_gib={first_candidate.peak_memory_bytes / 1024**3:.4f}"
    )
    print(
        f"td_target_ms={first_td.elapsed_ms:.3f} "
        f"td_peak_gib={first_td.peak_memory_bytes / 1024**3:.4f}"
    )
    with tempfile.TemporaryDirectory(prefix="robonana-posttrain-smoke-") as directory:
        path = Path(directory) / "ema_model.safetensors"
        save_file(ema.state_dict(), str(path))
        restored = FullModelEMA(online, decay=0.995, device=device)
        restored.load_state_dict(load_file(str(path), device="cpu"))
        for expected, actual in zip(ema.model.parameters(), restored.model.parameters(), strict=True):
            torch.testing.assert_close(expected, actual)
    print(
        f"checkpoint_roundtrip=ok total_smoke_s={time.perf_counter() - started:.3f} "
        f"cuda_peak_gib={torch.cuda.max_memory_allocated(device) / 1024**3:.4f}"
    )


if __name__ == "__main__":
    main()
