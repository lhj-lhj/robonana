"""FACT training-loop adapter for the shared FLUX.2 RoboNana model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from fact_train import Trainer
from flux2.model import Klein4BParams

# Imports register the raw HDF5 dataset and sampler with FACT.
from robonana.data import robotwin_hdf5 as _robotwin_hdf5  # noqa: F401
from robonana.models.pretrained import configure_trainable_parameters, load_flux2_fact_checkpoint
from robonana.sampling import flow_euler_schedule, flow_euler_step
from robonana.training.losses import joint_flow_loss
from robonana.training.visualization import (
    decode_flux2_tokens,
    log_pixel_eval,
    should_log_pixel_eval,
)


def _expand_timestep(timestep: Tensor, target: Tensor) -> Tensor:
    while timestep.ndim < target.ndim:
        timestep = timestep.unsqueeze(-1)
    return timestep.to(device=target.device, dtype=target.dtype)


def flow_noise(clean: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor]:
    noise = torch.randn_like(clean)
    sigma = _expand_timestep(timestep, clean)
    return clean * (1.0 - sigma) + noise * sigma, noise - clean


def text_position_ids(batch_size: int, length: int, device: torch.device) -> Tensor:
    ids = torch.zeros(batch_size, length, 4, device=device, dtype=torch.long)
    ids[:, :, 3] = torch.arange(length, device=device)
    return ids


def image_position_ids(
    batch_size: int,
    *,
    grid_height: int,
    grid_width: int,
    time_coord: Tensor,
    device: torch.device,
) -> Tensor:
    height = torch.arange(grid_height, device=device)
    width = torch.arange(grid_width, device=device)
    spatial = torch.cartesian_prod(height, width)
    ids = torch.zeros(batch_size, grid_height * grid_width, 4, device=device, dtype=torch.long)
    ids[:, :, 0] = time_coord.to(device=device, dtype=torch.long).reshape(batch_size, 1)
    ids[:, :, 1:3] = spatial[None]
    return ids


class RoboNanaTrainer(Trainer):
    """Reuse FACT's DataLoader, Accelerate, optimizer, checkpoint, and logging loop."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.memory_limit_gib = float(self.kwargs.get("memory_limit_gib", 0.0))
        if self.memory_limit_gib > 0 and self.device.type == "cuda":
            total_bytes = torch.cuda.get_device_properties(self.device).total_memory
            limit_bytes = int(self.memory_limit_gib * 1024**3)
            torch.cuda.set_per_process_memory_fraction(min(1.0, limit_bytes / total_bytes), self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.pixel_eval_interval = int(self.kwargs.get("pixel_eval_interval", 200))
        if self.pixel_eval_interval and self.pixel_eval_interval % self.log_interval:
            raise ValueError("pixel_eval_interval must be divisible by log_interval for atomic W&B logging")
        self.grid_height = int(self.kwargs.get("latent_grid_height", 12))
        self.grid_width = int(self.kwargs.get("latent_grid_width", 24))
        self.flow_shift = float(self.kwargs.get("flow_shift", 1.0))
        self.num_inference_steps = int(self.kwargs.get("num_inference_steps", 20))
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self.vae = None

    def get_models(self, model_config):
        checkpoint = str(model_config.checkpoint)
        action_dim = int(model_config.get("action_dim", 14))
        state_dim = int(model_config.get("state_dim", 14))
        max_horizon = int(model_config.get("max_horizon", 48))
        model, report = load_flux2_fact_checkpoint(
            checkpoint,
            action_dim=action_dim,
            state_dim=state_dim,
            max_horizon=max_horizon,
            device=self.device,
            dtype=self.dtype,
            params=Klein4BParams(),
        )
        train_mode = str(model_config.get("train_mode", "full"))
        trainable_names = configure_trainable_parameters(model, train_mode)
        if bool(model_config.get("gradient_checkpointing", True)):
            model.enable_gradient_checkpointing()
        model.train()
        self.model_name = "transformer"

        if self.is_main_process and self.pixel_eval_interval > 0:
            from diffusers.models import AutoencoderKLFlux2

            vae_dtype_name = str(model_config.get("vae_dtype", "float32"))
            vae_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[vae_dtype_name]
            self.vae = AutoencoderKLFlux2.from_pretrained(
                str(model_config.checkpoint_dir),
                subfolder="vae",
                torch_dtype=vae_dtype,
                local_files_only=True,
            ).eval()
            self.vae.requires_grad_(False)
            self.vae.to(self.device)

        if self.is_main_process:
            self.logger.info(
                "Loaded FLUX.2 backbone parameters=%d; trainable tensors=%d; pixel eval interval=%d",
                report.checkpoint_parameters,
                len(trainable_names),
                self.pixel_eval_interval,
            )
        return model

    def _sample_timestep(self, batch_size: int) -> Tensor:
        sigma = torch.rand(batch_size, device=self.device, dtype=torch.float32)
        if self.flow_shift != 1.0:
            sigma = self.flow_shift * sigma / (1.0 + (self.flow_shift - 1.0) * sigma)
        return sigma

    def save_checkpoint_step(self) -> None:
        if bool(self.kwargs.get("disable_checkpointing", False)):
            return
        super().save_checkpoint_step()

    def print_after_train(self) -> None:
        if self.device.type == "cuda":
            local_peak = torch.tensor(
                [torch.cuda.max_memory_allocated(self.device), torch.cuda.max_memory_reserved(self.device)],
                device=self.device,
                dtype=torch.float64,
            )
            all_peaks = self.accelerator.gather(local_peak).reshape(-1, 2)
            if self.is_main_process:
                peak = all_peaks.max(dim=0).values.cpu().tolist()
                self.logger.info(
                    "Peak CUDA memory across ranks: allocated=%.3f GiB, reserved=%.3f GiB, cap=%.3f GiB",
                    peak[0] / 1024**3,
                    peak[1] / 1024**3,
                    self.memory_limit_gib,
                )
        super().print_after_train()

    def _run_fixed_horizon_eval(
        self,
        *,
        batch_dict: dict[str, Any],
        context: Tensor,
        context_mask: Tensor,
        current: Tensor,
        state: Tensor,
        action: Tensor,
    ) -> None:
        horizons = batch_dict["eval_horizon_idx"][0].to(device=self.device, dtype=torch.long)
        count = horizons.numel()

        def repeat_first(value: Tensor) -> Tensor:
            return value[:1].expand(count, *value.shape[1:])

        eval_future = batch_dict["eval_future_latents"][0].to(device=self.device, dtype=self.dtype)
        eval_future_state = batch_dict["eval_future_state"][0].to(device=self.device, dtype=self.dtype).unsqueeze(1)
        eval_value = batch_dict["eval_value"][0].to(device=self.device, dtype=self.dtype).reshape(count, 1, 1)
        eval_context = repeat_first(context)
        eval_context_mask = repeat_first(context_mask)
        eval_current = repeat_first(current)
        eval_state = repeat_first(state)
        context_ids = text_position_ids(count, eval_context.shape[1], self.device)
        current_ids = image_position_ids(
            count,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            time_coord=torch.zeros_like(horizons),
            device=self.device,
        )
        future_ids = image_position_ids(
            count,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            time_coord=horizons,
            device=self.device,
        )
        schedule = flow_euler_schedule(
            self.num_inference_steps,
            flow_shift=self.flow_shift,
            device=self.device,
        )

        model_was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                # Stage 1 mirrors FACT inference: denoise action alone from pure
                # Gaussian noise. The attention mask makes the action sink
                # independent of every dummy suffix token supplied here.
                sampled_action = torch.randn_like(action[:1])
                dummy_gt_action = torch.zeros_like(sampled_action)
                dummy_future = torch.zeros_like(eval_future[:1])
                dummy_future_state = torch.zeros_like(eval_future_state[:1])
                dummy_value = torch.zeros_like(eval_value[:1])
                clean_time = torch.zeros(1, device=self.device, dtype=torch.float32)
                for sigma, sigma_next in zip(schedule[:-1], schedule[1:]):
                    action_time = sigma.expand(1)
                    action_output = self.model(
                        context=eval_context[:1],
                        context_ids=context_ids[:1],
                        current_latents=eval_current[:1],
                        current_ids=current_ids[:1],
                        noisy_future_latents=dummy_future,
                        future_ids=future_ids[:1],
                        state=eval_state[:1],
                        noisy_pred_action=sampled_action,
                        gt_action_cond=dummy_gt_action,
                        horizon_idx=horizons[:1],
                        noisy_future_state=dummy_future_state,
                        noisy_value=dummy_value,
                        action_timestep=action_time,
                        wm_timestep=clean_time,
                        context_mask=eval_context_mask[:1],
                    )
                    sampled_action = flow_euler_step(
                        sampled_action, action_output.action, sigma, sigma_next
                    )

                # Stage 2 uses the fully denoised Stage-1 action as the clean
                # teacher-forcing track and jointly denoises world targets.
                clean_action_cond = sampled_action.expand(count, -1, -1)
                pred_action_dummy = torch.zeros_like(clean_action_cond)
                sampled_future = torch.randn_like(eval_future[:1]).expand_as(eval_future).clone()
                sampled_future_state = (
                    torch.randn_like(eval_future_state[:1]).expand_as(eval_future_state).clone()
                )
                sampled_value = torch.randn_like(eval_value[:1]).expand_as(eval_value).clone()
                clean_action_time = torch.zeros(count, device=self.device, dtype=torch.float32)
                for sigma, sigma_next in zip(schedule[:-1], schedule[1:]):
                    wm_time = sigma.expand(count)
                    world_output = self.model(
                        context=eval_context,
                        context_ids=context_ids,
                        current_latents=eval_current,
                        current_ids=current_ids,
                        noisy_future_latents=sampled_future,
                        future_ids=future_ids,
                        state=eval_state,
                        noisy_pred_action=pred_action_dummy,
                        gt_action_cond=clean_action_cond,
                        horizon_idx=horizons,
                        noisy_future_state=sampled_future_state,
                        noisy_value=sampled_value,
                        action_timestep=clean_action_time,
                        wm_timestep=wm_time,
                        context_mask=eval_context_mask,
                    )
                    sampled_future = flow_euler_step(
                        sampled_future, world_output.image, sigma, sigma_next
                    )
                    sampled_future_state = flow_euler_step(
                        sampled_future_state, world_output.future_state, sigma, sigma_next
                    )
                    sampled_value = flow_euler_step(
                        sampled_value, world_output.value, sigma, sigma_next
                    )
        finally:
            if model_was_training:
                self.model.train()

        if self.vae is None:
            return

        def decode_each(tokens: Tensor) -> Tensor:
            return torch.cat(
                [
                    decode_flux2_tokens(
                        self.vae,
                        tokens[index : index + 1],
                        grid_height=self.grid_height,
                        grid_width=self.grid_width,
                    ).cpu()
                    for index in range(tokens.shape[0])
                ],
                dim=0,
            )

        with torch.no_grad():
            decoded_current = decode_each(current[:1])
            decoded_targets = decode_each(eval_future)
            decoded_predictions = decode_each(sampled_future)
        log_pixel_eval(
            accelerator=self.accelerator,
            step=self.cur_step,
            current=decoded_current,
            targets=decoded_targets,
            predictions=decoded_predictions,
            horizons=horizons,
            num_inference_steps=self.num_inference_steps,
        )

    def forward_step(self, batch_dict: dict[str, Any]):
        context = batch_dict["context"].to(device=self.device, dtype=self.dtype)
        current = batch_dict["current_latents"].to(device=self.device, dtype=self.dtype)
        future = batch_dict["future_latents"].to(device=self.device, dtype=self.dtype)
        state = batch_dict["state"].to(device=self.device, dtype=self.dtype).unsqueeze(1)
        action = batch_dict["action"].to(device=self.device, dtype=self.dtype)
        future_state = batch_dict["future_state"].to(device=self.device, dtype=self.dtype).unsqueeze(1)
        value = batch_dict["value"].to(device=self.device, dtype=self.dtype).reshape(context.shape[0], 1, 1)
        horizon = batch_dict["horizon_idx"].to(device=self.device, dtype=torch.long).reshape(-1)
        context_mask = batch_dict["context_mask"].to(device=self.device, dtype=torch.bool)
        action_loss_mask = batch_dict["action_loss_mask"].to(device=self.device)

        batch_size = context.shape[0]
        expected_tokens = self.grid_height * self.grid_width
        if current.shape[1] != expected_tokens or future.shape[1] != expected_tokens:
            raise ValueError(
                f"cached FLUX image tokens must use {self.grid_height}x{self.grid_width}={expected_tokens} tokens"
            )
        action_timestep = self._sample_timestep(batch_size)
        wm_timestep = self._sample_timestep(batch_size)
        noisy_action, action_target = flow_noise(action, action_timestep)
        noisy_future, image_target = flow_noise(future, wm_timestep)
        noisy_future_state, future_state_target = flow_noise(future_state, wm_timestep)
        noisy_value, value_target = flow_noise(value, wm_timestep)

        context_ids = text_position_ids(batch_size, context.shape[1], self.device)
        current_ids = image_position_ids(
            batch_size,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            time_coord=torch.zeros_like(horizon),
            device=self.device,
        )
        future_ids = image_position_ids(
            batch_size,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            time_coord=horizon,
            device=self.device,
        )

        output = self.model(
            context=context,
            context_ids=context_ids,
            current_latents=current,
            current_ids=current_ids,
            noisy_future_latents=noisy_future,
            future_ids=future_ids,
            state=state,
            noisy_pred_action=noisy_action,
            gt_action_cond=action,
            horizon_idx=horizon,
            noisy_future_state=noisy_future_state,
            noisy_value=noisy_value,
            action_timestep=action_timestep,
            wm_timestep=wm_timestep,
            context_mask=context_mask,
        )

        if should_log_pixel_eval(self.cur_step, self.pixel_eval_interval):
            self._run_fixed_horizon_eval(
                batch_dict=batch_dict,
                context=context,
                context_mask=context_mask,
                current=current,
                state=state,
                action=action,
            )

        return joint_flow_loss(
            output,
            image_target=image_target,
            action_target=action_target,
            future_state_target=future_state_target,
            value_target=value_target,
            action_loss_mask=action_loss_mask,
        )

    def parse_losses(self, losses: dict[str, Tensor] | Tensor) -> Tensor:
        if not isinstance(losses, dict):
            return super().parse_losses(losses)
        weights = dict(self.kwargs.get("loss_weights", {}))
        reduced = {key: value.mean() for key, value in losses.items()}
        loss = sum(value * float(weights.get(key, 1.0)) for key, value in reduced.items())
        gathered = {key: self.accelerator.gather(value).mean() for key, value in reduced.items()}
        total_loss = sum(value * float(weights.get(key, 1.0)) for key, value in gathered.items())
        outputs = {**gathered, "total_loss": total_loss}
        if torch.isnan(total_loss).any():
            loss = torch.full((), float("nan"), device=loss.device)
        loss_nan_total_limit = int(self.kwargs.get("loss_nan_total_limit", 100))
        if torch.isnan(loss).any():
            self._loss_nan_count += 1
            if loss_nan_total_limit > 0 and self._loss_nan_count > loss_nan_total_limit:
                raise RuntimeError("loss remained NaN beyond loss_nan_total_limit")
        else:
            self._loss_nan_count = 0
        for key, value in outputs.items():
            if key not in self._outputs:
                self._outputs[key] = {"sum": 0.0, "num": 0}
            self._outputs[key]["sum"] += float(value.detach().item())
            self._outputs[key]["num"] += 1
        return loss
