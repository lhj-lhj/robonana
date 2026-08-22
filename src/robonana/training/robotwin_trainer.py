"""FACT training-loop adapter for the shared FLUX.2 RoboNana model."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from fact_train import Trainer
from flux2.model import Flux2Params, Klein4BParams

# Imports register the raw HDF5 dataset and sampler with FACT.
from robonana.data import robotwin_hdf5 as _robotwin_hdf5  # noqa: F401
from robonana.models.pretrained import (
    configure_trainable_parameters,
    initialize_flux2_fact_model,
    load_flux2_fact_checkpoint,
)
from robonana.models.position_ids import image_position_ids, text_position_ids
from robonana.sampling import flow_euler_schedule, sample_two_stage_flow, sample_world_flow
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


def resolve_cuda_device_index(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return device.index if device.index is not None else torch.cuda.current_device()


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read both FACT Config attributes and ordinary mapping keys reliably."""
    try:
        return getattr(config, name)
    except AttributeError:
        if isinstance(config, Mapping):
            return config.get(name, default)
        getter = getattr(config, "get", None)
        return getter(name, default) if getter is not None else default


class RoboNanaTrainer(Trainer):
    """Reuse FACT's DataLoader, Accelerate, optimizer, checkpoint, and logging loop."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.memory_limit_gib = float(self.kwargs.get("memory_limit_gib", 0.0))
        self.cuda_device_index = resolve_cuda_device_index(self.device)
        if self.memory_limit_gib > 0 and self.device.type == "cuda":
            total_bytes = torch.cuda.get_device_properties(self.cuda_device_index).total_memory
            limit_bytes = int(self.memory_limit_gib * 1024**3)
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, limit_bytes / total_bytes), self.cuda_device_index
            )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.cuda_device_index)
        self.pixel_eval_interval = int(self.kwargs.get("pixel_eval_interval", 200))
        if self.pixel_eval_interval and self.pixel_eval_interval % self.log_interval:
            raise ValueError("pixel_eval_interval must be divisible by log_interval for atomic W&B logging")
        self.grid_height = int(self.kwargs.get("latent_grid_height", 12))
        self.grid_width = int(self.kwargs.get("latent_grid_width", 24))
        self.flow_shift = float(self.kwargs.get("flow_shift", 1.0))
        self.num_inference_steps = int(self.kwargs.get("num_inference_steps", 20))
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self._pending_pixel_eval: dict[str, Tensor] | None = None
        self._optimizer_step_succeeded = False
        self.vae_checkpoint_dir: str | None = None
        self.vae_dtype = torch.float32

    def get_models(self, model_config):
        action_dim = int(_config_value(model_config, "action_dim", 14))
        state_dim = int(_config_value(model_config, "state_dim", 14))
        max_horizon = int(_config_value(model_config, "max_horizon", 48))
        params_config = _config_value(model_config, "params", None)
        params = Klein4BParams() if params_config is None else Flux2Params(**dict(params_config))
        checkpoint = _config_value(model_config, "checkpoint", None)
        initialization = str(
            _config_value(
                model_config,
                "initialization",
                "pretrained" if checkpoint is not None else "scratch",
            )
        )
        if initialization == "pretrained":
            if checkpoint is None:
                raise ValueError("pretrained initialization requires models.checkpoint")
            model, report = load_flux2_fact_checkpoint(
                str(checkpoint),
                action_dim=action_dim,
                state_dim=state_dim,
                max_horizon=max_horizon,
                device=self.device,
                dtype=self.dtype,
                params=params,
            )
            initialization_label = f"pretrained checkpoint parameters={report.checkpoint_parameters}"
        elif initialization == "scratch":
            model = initialize_flux2_fact_model(
                action_dim=action_dim,
                state_dim=state_dim,
                max_horizon=max_horizon,
                device=self.device,
                dtype=self.dtype,
                params=params,
            )
            initialization_label = "scratch"
        else:
            raise ValueError(f"initialization must be 'pretrained' or 'scratch', got {initialization!r}")
        train_mode = str(_config_value(model_config, "train_mode", "full"))
        trainable_names = configure_trainable_parameters(model, train_mode)
        if bool(_config_value(model_config, "gradient_checkpointing", True)):
            model.enable_gradient_checkpointing()
        else:
            model.disable_gradient_checkpointing()
        model.train()
        self.model_name = "transformer"

        self.vae_checkpoint_dir = str(_config_value(model_config, "checkpoint_dir"))
        vae_dtype_name = str(_config_value(model_config, "vae_dtype", "float32"))
        try:
            self.vae_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[vae_dtype_name]
        except KeyError as error:
            raise ValueError(f"unsupported VAE dtype: {vae_dtype_name}") from error

        if self.is_main_process:
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            self.logger.info(
                "Initialized FLUX.2 backbone=%s; parameters=%d; trainable_parameters=%d; "
                "trainable_tensors=%d; gradient_checkpointing=%s; pixel_eval_interval=%d",
                initialization_label,
                parameter_count,
                trainable_count,
                len(trainable_names),
                model.gradient_checkpointing,
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
        early_steps = {int(step) for step in self.kwargs.get("early_checkpoint_steps", ())}
        if self.cur_step in early_steps and self.cur_step % int(self.checkpoint_interval):
            checkpoint_interval = self.checkpoint_interval
            try:
                self.checkpoint_interval = 1
                super().save_checkpoint_step()
            finally:
                self.checkpoint_interval = checkpoint_interval
            return
        super().save_checkpoint_step()

    def backward_step(self, loss: Tensor) -> None:
        loss_is_finite = bool(torch.isfinite(loss.detach()).all().item())
        super().backward_step(loss)
        self._optimizer_step_succeeded = loss_is_finite and self.accelerator.sync_gradients

    def print_step(self) -> None:
        pending_eval = self._pending_pixel_eval
        self._pending_pixel_eval = None
        if self._optimizer_step_succeeded and pending_eval is not None:
            self._run_fixed_horizon_eval(pending_eval)
        self._optimizer_step_succeeded = False
        super().print_step()

    def print_after_train(self) -> None:
        if self.device.type == "cuda":
            local_peak = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(self.cuda_device_index),
                    torch.cuda.max_memory_reserved(self.cuda_device_index),
                ],
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

    def _stage_fixed_horizon_eval(
        self,
        *,
        batch_dict: dict[str, Any],
        context: Tensor,
        context_mask: Tensor,
        current: Tensor,
        state: Tensor,
        action: Tensor,
    ) -> None:
        self._pending_pixel_eval = {
            "sample_index": batch_dict["sample_index"][0].detach().cpu(),
            "context": context[:1].detach(),
            "context_mask": context_mask[:1].detach(),
            "current": current[:1].detach(),
            "state": state[:1].detach(),
            "action": action[:1].detach(),
        }

    def _run_fixed_horizon_eval(self, payload: dict[str, Tensor]) -> None:
        dataset = self.dataloader.dataset
        while not hasattr(dataset, "load_eval_future_latents"):
            if hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            elif getattr(dataset, "datasets", None):
                # Pixel monitoring always uses the clean initial-data child.
                dataset = dataset.datasets[0]
            else:
                break
        if not hasattr(dataset, "load_eval_future_latents") or not hasattr(dataset, "eval_horizons"):
            raise TypeError("pixel eval requires RoboTwinHDF5Dataset eval accessors")
        horizons = torch.tensor(dataset.eval_horizons, device=self.device, dtype=torch.long)
        count = horizons.numel()
        if self.is_main_process:
            self.logger.info(
                "Start post-optimizer pixel eval: ranks=%d, horizons=%s, inference_steps=%d",
                self.accelerator.num_processes,
                horizons.detach().cpu().tolist(),
                self.num_inference_steps,
            )

        def repeat_first(value: Tensor) -> Tensor:
            return value[:1].expand(count, *value.shape[1:])

        eval_context = repeat_first(payload["context"])
        eval_context_mask = repeat_first(payload["context_mask"])
        eval_current = repeat_first(payload["current"])
        eval_state = repeat_first(payload["state"])
        action = payload["action"]
        future_template = eval_current
        future_state_template = eval_state
        value_template = torch.empty(count, 1, 1, device=self.device, dtype=self.dtype)
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
            with torch.inference_mode():
                # Stage 1 mirrors FACT inference: denoise action alone from pure
                # Gaussian noise. The attention mask makes the action sink
                # independent of every dummy suffix token supplied here.
                action_noise = torch.randn_like(action[:1])
                dummy_gt_action = torch.zeros_like(action_noise)
                dummy_future = torch.zeros_like(future_template[:1])
                dummy_future_state = torch.zeros_like(future_state_template[:1])
                dummy_value = torch.zeros_like(value_template[:1])
                clean_time = torch.zeros(1, device=self.device, dtype=torch.float32)

                def predict_action(sampled_action: Tensor, sigma: Tensor) -> Tensor:
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
                    return action_output.action

                # Stage 2 uses the fully denoised Stage-1 action as the clean
                # teacher-forcing track and jointly denoises world targets.
                future_noise = torch.randn_like(future_template)
                future_state_noise = torch.randn_like(future_state_template)
                value_noise = torch.randn_like(value_template)
                clean_action_time = torch.zeros(count, device=self.device, dtype=torch.float32)

                def predict_world(
                    sampled_future: Tensor,
                    sampled_future_state: Tensor,
                    sampled_value: Tensor,
                    sampled_action: Tensor,
                    sigma: Tensor,
                ) -> tuple[Tensor, Tensor, Tensor]:
                    clean_action_cond = sampled_action.expand(count, -1, -1)
                    pred_action_dummy = torch.zeros_like(clean_action_cond)
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
                    return world_output.image, world_output.future_state, world_output.value

                samples = sample_two_stage_flow(
                    action_noise=action_noise,
                    future_noise=future_noise,
                    future_state_noise=future_state_noise,
                    value_noise=value_noise,
                    schedule=schedule,
                    predict_action=predict_action,
                    predict_world=predict_world,
                )
                gt_action_samples = sample_world_flow(
                    clean_action=action[:1].expand(count, -1, -1),
                    future_noise=future_noise,
                    future_state_noise=future_state_noise,
                    value_noise=value_noise,
                    schedule=schedule,
                    predict_world=predict_world,
                )
        finally:
            if model_was_training:
                self.model.train()

        # Ground-truth future images are visualization-only. They are loaded
        # after pure-noise inference and only on periodic eval steps.
        eval_future = dataset.load_eval_future_latents(
            int(payload["sample_index"].item()),
            horizons.detach().cpu().tolist(),
        ).to(device=self.device, dtype=self.dtype)
        if self.is_main_process:
            self.logger.info("Lazily loaded GT future latents after pure-noise sampling")

        if self.vae_checkpoint_dir is None:
            raise RuntimeError("VAE checkpoint directory was not configured")
        from diffusers.models import AutoencoderKLFlux2

        if self.is_main_process:
            self.logger.info("Load FP32 FLUX.2 VAE on every rank for local pixel eval decode")
        vae = AutoencoderKLFlux2.from_pretrained(
            self.vae_checkpoint_dir,
            subfolder="vae",
            torch_dtype=self.vae_dtype,
            local_files_only=True,
        ).eval()
        vae.requires_grad_(False)
        vae.to(self.device)
        try:
            with torch.inference_mode():
                local_current = decode_flux2_tokens(
                    vae,
                    payload["current"],
                    grid_height=self.grid_height,
                    grid_width=self.grid_width,
                )
                local_targets = decode_flux2_tokens(
                    vae,
                    eval_future,
                    grid_height=self.grid_height,
                    grid_width=self.grid_width,
                )
                local_predictions = decode_flux2_tokens(
                    vae,
                    samples.future,
                    grid_height=self.grid_height,
                    grid_width=self.grid_width,
                )
                local_gt_action_predictions = decode_flux2_tokens(
                    vae,
                    gt_action_samples.future,
                    grid_height=self.grid_height,
                    grid_width=self.grid_width,
                )

                def to_uint8(images: Tensor) -> Tensor:
                    return images.mul(255).round().to(torch.uint8)

                local_current = to_uint8(local_current)
                local_targets = to_uint8(local_targets)
                local_predictions = to_uint8(local_predictions)
                local_gt_action_predictions = to_uint8(local_gt_action_predictions)
        finally:
            vae.to("cpu")
            del vae
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        gathered_current = self.accelerator.gather(local_current)
        gathered_targets = self.accelerator.gather(local_targets).reshape(
            self.accelerator.num_processes, count, *local_targets.shape[1:]
        )
        gathered_predictions = self.accelerator.gather(local_predictions).reshape(
            self.accelerator.num_processes, count, *local_predictions.shape[1:]
        )
        gathered_gt_action_predictions = self.accelerator.gather(
            local_gt_action_predictions
        ).reshape(
            self.accelerator.num_processes,
            count,
            *local_gt_action_predictions.shape[1:],
        )
        gathered_horizons = self.accelerator.gather(horizons.unsqueeze(0))

        try:
            if self.is_main_process:
                decoded_current = gathered_current.float().div(255).cpu()
                decoded_targets = gathered_targets.float().div(255).cpu()
                decoded_predictions = gathered_predictions.float().div(255).cpu()
                decoded_gt_action_predictions = (
                    gathered_gt_action_predictions.float().div(255).cpu()
                )
                log_pixel_eval(
                    accelerator=self.accelerator,
                    step=self.cur_step,
                    current=decoded_current,
                    targets=decoded_targets,
                    predictions=decoded_predictions,
                    gt_action_predictions=decoded_gt_action_predictions,
                    horizons=gathered_horizons,
                    num_inference_steps=self.num_inference_steps,
                )
                self.logger.info(
                    "Gathered %d locally decoded pixel rows on rank 0 for W&B",
                    self.accelerator.num_processes,
                )
        finally:
            self.accelerator.wait_for_everyone()
        if self.is_main_process:
            self.logger.info("Removed FLUX.2 VAE from every rank GPU after pixel eval")

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
            self._stage_fixed_horizon_eval(
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
