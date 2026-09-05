"""FACT training-loop adapter for the shared FLUX.2 RoboNana model."""

from __future__ import annotations

import copy
import gc
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from fact_train import Trainer, build_optimizer
from fact_train.utils import as_list
from flux2.model import Flux2Params
from world_action_model.image_layouts import ROBOTWIN_VIEW_KEYS

# Imports register the raw HDF5 dataset and sampler with FACT.
from robonana.data import robotwin_hdf5 as _robotwin_hdf5  # noqa: F401
from robonana.encoding import DinoV3FeatureEncoder
from robonana.models.pretrained import (
    configure_trainable_parameters,
    initialize_flux2_fact_model,
    load_flux2_fact_checkpoint,
    load_flux2_fact_trained_checkpoint,
    load_mac_from_legacy_checkpoint,
)
from robonana.models.position_ids import dino_position_ids, image_position_ids, text_position_ids
from robonana.sampling import (
    evaluate_mac_critics,
    flow_euler_schedule,
    generate_mac_imaginary_rollout_h1,
    sample_world_flow,
)
from robonana.training.losses import (
    deterministic_return_loss,
    joint_flow_loss,
    masked_bce_with_logits,
    masked_elementwise_bce_with_logits,
    masked_mse,
)
from robonana.training.optimizer import build_optimizer_param_groups
from robonana.training.posttraining import (
    ValueExpertEMA,
    evaluating,
)
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


def _validate_initial_global_step(initial_step: int, max_steps: int) -> int:
    initial_step = int(initial_step)
    max_steps = int(max_steps)
    if initial_step < 0:
        raise ValueError("initial_global_step cannot be negative")
    if initial_step >= max_steps:
        raise ValueError(
            "initial_global_step must be smaller than max_steps, got "
            f"{initial_step} >= {max_steps}"
        )
    return initial_step


class RoboNanaTrainer(Trainer):
    """Reuse FACT's DataLoader, Accelerate, optimizer, checkpoint, and logging loop."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        initial_global_step = int(self.kwargs.get("initial_global_step", 0))
        if initial_global_step:
            initial_global_step = _validate_initial_global_step(
                initial_global_step, self._max_steps
            )
            if self.cur_step != 0:
                raise RuntimeError(
                    "FACT initialized the trainer at a nonzero step unexpectedly"
                )
            self._cur_step = initial_global_step
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
        self._accumulation_invalid = False
        self.vae_checkpoint_dir: str | None = None
        self.vae_dtype = torch.float32
        self.dino_dim: int | None = None
        self.dino_encoder: DinoV3FeatureEncoder | None = None
        self.dino_encoder_batch_size = 0
        self.posttrain_config = dict(self.kwargs.get("posttrain", {}))
        self.posttrain_enabled = bool(self.posttrain_config.get("enabled", False))
        self.posttrain_q_target_mode = str(
            self.posttrain_config.get(
                "q_target_mode", self.kwargs.get("q_target_mode", "")
            )
        )
        self.mac_enabled = self.posttrain_q_target_mode == "mac_mot_v2"
        self.mac_phase = str(self.posttrain_config.get("phase", "world_policy"))
        self.target_value_ema: ValueExpertEMA | None = None
        self.ema_forward_autocast_dtype = torch.bfloat16
        self.current_collection_round = int(
            self.posttrain_config.get("current_collection_round", 0)
        )
        self._posttrain_metrics: dict[str, Tensor] = {}
        if self.posttrain_enabled:
            self._validate_posttrain_config()

    def _validate_posttrain_config(self) -> None:
        configured_mode = str(self.kwargs.get("q_target_mode", ""))
        if configured_mode != "mac_mot_v2":
            raise ValueError(
                "the maintained RL path is mac_mot_v2; legacy TD/MC posttraining was removed"
            )
        if configured_mode != self.posttrain_q_target_mode:
            raise ValueError("train and posttrain q_target_mode must match")
        if self.posttrain_config.get("algorithm") != "mac_mot_v2":
            raise ValueError("mac_mot_v2 posttraining requires algorithm='mac_mot_v2'")
        if self.mac_phase not in {"world_policy", "critic"}:
            raise ValueError("mac_mot_v2 phase must be world_policy or critic")
        if int(self.posttrain_config.get("chunk_horizon", 0)) != 48:
            raise ValueError("mac_mot_v2 requires chunk_horizon=48")
        imagination = dict(self.posttrain_config.get("imagination", {}))
        if int(imagination.get("rollout_chunks", 0)) != 1:
            raise ValueError("mac_mot_v2 supports exactly one imaginary rollout chunk")
        if int(imagination.get("candidate_count", 0)) <= 0:
            raise ValueError("mac_mot_v2 candidate_count must be positive")
        if imagination.get("candidate_selection") != "argmax_q":
            raise ValueError("mac_mot_v2 candidate selection must be argmax_q")
        if imagination.get("fresh_each_batch") is not True:
            raise ValueError("mac_mot_v2 requires a fresh imaginary rollout per batch")
        if imagination.get("stop_gradient_target") is not True:
            raise ValueError("mac_mot_v2 critic targets must be stop-gradient")
        if float(self.posttrain_config.get("return_scale", 0.0)) <= 0:
            raise ValueError("mac_mot_v2 return_scale must be positive")
        ema = dict(self.posttrain_config.get("ema", {}))
        if ema.get("storage_dtype") != "float32":
            raise ValueError("target Value EMA storage_dtype must be float32")
        if ema.get("forward_autocast_dtype") != "bfloat16":
            raise ValueError("target Value forward dtype must be bfloat16")
        if ema.get("target") != "value_expert_only":
            raise ValueError("mac_mot_v2 EMA target must be value_expert_only")

    def set_ema_models(self) -> None:
        if not self.mac_enabled:
            return super().set_ema_models()
        if self.with_ema:
            raise ValueError("disable FACT EMA for mac_mot_v2")
        if self.mac_phase == "world_policy":
            return
        if len(self.models) != 1:
            raise ValueError("mac_mot_v2 requires one shared model")
        ema = dict(self.posttrain_config["ema"])
        self.target_value_ema = ValueExpertEMA(
            self.models[0].value_expert,
            decay=float(ema["decay"]),
            update_every_optimizer_steps=int(ema["update_every_optimizer_steps"]),
            start_step=int(ema["start_step"]),
            device=self.device,
        )
        initial_checkpoint = str(ema.get("initial_checkpoint", "")).strip()
        initial_state = str(ema.get("initial_state", "")).strip()
        if initial_checkpoint:
            target_path = Path(initial_checkpoint).expanduser()
            if not target_path.is_file():
                raise FileNotFoundError(
                    f"target Value initialization checkpoint not found: {target_path}"
                )
            self.target_value_ema.load_state_dict(
                load_file(str(target_path), device="cpu")
            )
            if initial_state:
                state_path = Path(initial_state).expanduser()
                if not state_path.is_file():
                    raise FileNotFoundError(
                        f"target Value initialization state not found: {state_path}"
                    )
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.target_value_ema.update_count = int(state.get("update_count", 0))

    def prepare(self, dataloaders: Any, models: Any, optimizers: Any, schedulers: Any) -> None:
        super().prepare(dataloaders, models, optimizers, schedulers)
        if self.target_value_ema is not None:
            for optimizer in self.optimizers:
                self.target_value_ema.assert_not_in_optimizer(optimizer)
            if self.is_main_process:
                self.logger.info(
                    "Initialized target Value expert only: decay=%.6f parameters=%d; "
                    "FLUX copies=0 Q-target copies=0",
                    self.target_value_ema.decay,
                    sum(parameter.numel() for parameter in self.target_value_ema.model.parameters()),
                )

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        if self.posttrain_enabled:
            state.update(
                ema_update_count=(
                    0 if self.target_value_ema is None else self.target_value_ema.update_count
                ),
                current_collection_round=self.current_collection_round,
                posttrain_config=self.posttrain_config,
            )
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if not self.posttrain_enabled:
            return
        self.current_collection_round = int(
            state_dict.get("current_collection_round", self.current_collection_round)
        )
        if self.target_value_ema is not None:
            self.target_value_ema.update_count = int(state_dict.get("ema_update_count", 0))

    def get_models(self, model_config):
        action_dim = int(_config_value(model_config, "action_dim", 14))
        state_dim = int(_config_value(model_config, "state_dim", 14))
        reward_dim = int(_config_value(model_config, "reward_dim", 1))
        success_dim = int(_config_value(model_config, "success_dim", 1))
        q_dim = int(_config_value(model_config, "q_dim", 1))
        reward_head_type = str(_config_value(model_config, "reward_head_type", "direct"))
        max_horizon = int(_config_value(model_config, "max_horizon", 48))
        architecture_version = str(
            _config_value(model_config, "architecture_version", "legacy_v1")
        )
        chunk_horizon = int(_config_value(model_config, "chunk_horizon", max_horizon))
        value_dim = int(_config_value(model_config, "value_dim", 1))
        expert_hidden_dim = _config_value(model_config, "expert_hidden_dim", None)
        expert_hidden_dim = None if expert_hidden_dim is None else int(expert_hidden_dim)
        if architecture_version == "mac_mot_v2":
            if reward_head_type != "binary_chunk" or reward_dim != chunk_horizon:
                raise ValueError(
                    "mac_mot_v2 requires reward_head_type='binary_chunk' and reward_dim=chunk_horizon"
                )
        elif reward_head_type != "direct" or success_dim != 1:
            raise ValueError(
                "current training requires models.reward_head_type='direct' and success_dim=1"
            )
        raw_dino_dim = _config_value(model_config, "dino_dim", None)
        dino_dim = None if raw_dino_dim is None else int(raw_dino_dim)
        pred_action_bidirectional = _config_value(
            model_config, "pred_action_bidirectional", False
        )
        if not isinstance(pred_action_bidirectional, bool):
            raise TypeError("models.pred_action_bidirectional must be a bool")
        self.dino_dim = dino_dim
        if dino_dim is not None:
            if dino_dim != 3072:
                raise ValueError(f"online DINOv3 ViT-B/16 requires dino_dim=3072, got {dino_dim}")
            self.dino_encoder_batch_size = int(
                _config_value(model_config, "dino_encoder_batch_size", 96)
            )
            if self.dino_encoder_batch_size <= 0:
                raise ValueError("models.dino_encoder_batch_size must be positive")
            self.dino_encoder = DinoV3FeatureEncoder(
                str(
                    _config_value(
                        model_config,
                        "dino_encoder_model",
                        "vit_base_patch16_dinov3.lvd1689m",
                    )
                ),
                device=self.device,
                dtype=self.dtype,
            )
        params_config = _config_value(model_config, "params", None)
        if params_config is None:
            raise ValueError("models.params must record the complete FLUX.2 architecture")
        params = Flux2Params(**dict(params_config))
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
                reward_dim=reward_dim,
                success_dim=success_dim,
                q_dim=q_dim,
                max_horizon=max_horizon,
                dino_dim=dino_dim,
                pred_action_bidirectional=pred_action_bidirectional,
                device=self.device,
                dtype=self.dtype,
                params=params,
                architecture_version=architecture_version,
                chunk_horizon=chunk_horizon,
                value_dim=value_dim,
                expert_hidden_dim=expert_hidden_dim,
            )
            initialization_label = f"pretrained checkpoint parameters={report.checkpoint_parameters}"
        elif initialization == "trained":
            if checkpoint is None:
                raise ValueError("trained initialization requires models.checkpoint")
            model, report = load_flux2_fact_trained_checkpoint(
                str(checkpoint),
                action_dim=action_dim,
                state_dim=state_dim,
                reward_dim=reward_dim,
                success_dim=success_dim,
                q_dim=q_dim,
                reward_head_type=reward_head_type,
                max_horizon=max_horizon,
                dino_dim=dino_dim,
                pred_action_bidirectional=pred_action_bidirectional,
                architecture_version=architecture_version,
                chunk_horizon=chunk_horizon,
                value_dim=value_dim,
                expert_hidden_dim=expert_hidden_dim,
                device=self.device,
                dtype=self.dtype,
                params=params,
                config_path=_config_value(model_config, "checkpoint_config", None),
            )
            initialization_label = (
                f"trained checkpoint parameters={report.checkpoint_parameters}; "
                f"new_parameters={len(report.initialized_robot_parameters)}"
            )
        elif initialization == "mac_from_legacy":
            if checkpoint is None:
                raise ValueError("mac_from_legacy initialization requires models.checkpoint")
            checkpoint_config = _config_value(model_config, "checkpoint_config", None)
            if checkpoint_config is None:
                raise ValueError(
                    "mac_from_legacy initialization requires the exact 120k checkpoint_config"
                )
            model, report = load_mac_from_legacy_checkpoint(
                str(checkpoint),
                config_path=str(checkpoint_config),
                action_dim=action_dim,
                state_dim=state_dim,
                reward_dim=reward_dim,
                success_dim=success_dim,
                q_dim=q_dim,
                value_dim=value_dim,
                expert_hidden_dim=1024 if expert_hidden_dim is None else expert_hidden_dim,
                chunk_horizon=chunk_horizon,
                dino_dim=dino_dim,
                device=self.device,
                dtype=self.dtype,
                params=params,
            )
            initialization_label = (
                f"120k MAC migration loaded={len(report.loaded_parameter_names)}; "
                f"skipped={len(report.skipped_checkpoint_parameters)}; "
                f"new={len(report.initialized_robot_parameters)}"
            )
        elif initialization == "scratch":
            model = initialize_flux2_fact_model(
                action_dim=action_dim,
                state_dim=state_dim,
                reward_dim=reward_dim,
                success_dim=success_dim,
                q_dim=q_dim,
                max_horizon=max_horizon,
                dino_dim=dino_dim,
                pred_action_bidirectional=pred_action_bidirectional,
                device=self.device,
                dtype=self.dtype,
                params=params,
                architecture_version=architecture_version,
                chunk_horizon=chunk_horizon,
                value_dim=value_dim,
                expert_hidden_dim=expert_hidden_dim,
            )
            initialization_label = "scratch"
        else:
            raise ValueError(
                "initialization must be pretrained, trained, mac_from_legacy, or scratch, "
                f"got {initialization!r}"
            )
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
            self.logger.info(
                "Attention layout: architecture=%s; A=%s; clean_action=%s",
                architecture_version,
                "bidirectional" if model.pred_action_bidirectional else "causal",
                "full-48" if architecture_version == "mac_mot_v2" else "causal-prefix",
            )
            if self.dino_encoder is not None:
                self.logger.info(
                    "Frozen online DINO encoder=%s; inference_batch_size=%d; checkpoint_excluded=true",
                    self.dino_encoder.model_name,
                    self.dino_encoder_batch_size,
                )
        return model

    def get_optimizers(self, optimizers):
        optimizer_configs = as_list(optimizers)
        if not any(isinstance(config, dict) and "robot_lr" in config for config in optimizer_configs):
            return super().get_optimizers(optimizers)
        if len(optimizer_configs) != 1 or len(self.models) != 1:
            raise ValueError("robot_lr requires exactly one optimizer and one model")

        optimizer_config = copy.deepcopy(optimizer_configs[0])
        robot_lr = float(optimizer_config.pop("robot_lr"))
        base_lr = float(optimizer_config["lr"])
        param_groups = build_optimizer_param_groups(
            self.models[0],
            base_lr=base_lr,
            robot_lr=robot_lr,
        )
        if not param_groups:
            raise ValueError("optimizer has no trainable parameters")
        optimizer = build_optimizer(optimizer_config, params=param_groups)
        if self.is_main_process:
            for group in optimizer.param_groups:
                self.logger.info(
                    "Optimizer group %s: lr=%.2e tensors=%d params=%d",
                    group["name"],
                    float(group["lr"]),
                    len(group["params"]),
                    sum(int(parameter.numel()) for parameter in group["params"]),
                )
        return [optimizer]

    def save_model_hook(self, models, weights, output_dir: str) -> None:
        super().save_model_hook(models, weights, output_dir)
        if self.target_value_ema is not None and self.is_main_process:
            output = Path(output_dir)
            save_file(
                self.target_value_ema.state_dict(),
                str(output / "target_value_expert.safetensors"),
            )
            state = {
                "decay": self.target_value_ema.decay,
                "update_every_optimizer_steps": self.target_value_ema.update_every_optimizer_steps,
                "start_step": self.target_value_ema.start_step,
                "update_count": self.target_value_ema.update_count,
                "storage_dtype": "float32",
                "target": "value_expert_only",
                "current_collection_round": self.current_collection_round,
            }
            (output / "value_ema_state.json").write_text(
                json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (output / "posttrain_config.json").write_text(
                json.dumps(self.posttrain_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return

    def load_model_hook(self, models, input_dir: str) -> None:
        super().load_model_hook(models, input_dir)
        if self.target_value_ema is not None:
            target_path = Path(input_dir) / "target_value_expert.safetensors"
            state_path = Path(input_dir) / "value_ema_state.json"
            if target_path.is_file():
                self.target_value_ema.load_state_dict(
                    load_file(str(target_path), device="cpu")
                )
                if state_path.is_file():
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    self.target_value_ema.update_count = int(state.get("update_count", 0))
                    self.current_collection_round = int(
                        state.get("current_collection_round", self.current_collection_round)
                    )
                source = str(target_path)
            else:
                online = self.accelerator.unwrap_model(
                    self.model, keep_torch_compile=False
                )
                self.target_value_ema.exact_copy_from(online.value_expert)
                self.target_value_ema.update_count = 0
                source = "exact-copy online Value (new critic phase)"
            for optimizer in self.optimizers:
                self.target_value_ema.assert_not_in_optimizer(optimizer)
            if self.is_main_process:
                self.logger.info(
                    "Restored target Value expert from %s: decay=%.6f updates=%d",
                    source,
                    self.target_value_ema.decay,
                    self.target_value_ema.update_count,
                )
            return

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
        finite_flag = torch.isfinite(loss.detach()).all().to(
            device=loss.device, dtype=torch.int32
        )
        if int(getattr(self.accelerator, "num_processes", 1)) > 1:
            finite_flag = self.accelerator.reduce(finite_flag, reduction="min")
        loss_is_finite = bool(finite_flag.item())
        if not loss_is_finite:
            self._accumulation_invalid = True
        if bool(getattr(self, "_accumulation_invalid", False)):
            self._optimizer_step_succeeded = False
            # One bad accumulation micro-step invalidates the whole optimizer
            # step on every rank.  Clear partial gradients at the synchronization
            # boundary and deliberately do not advance the scheduler or EMA.
            if self.accelerator.sync_gradients:
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                self._accumulation_invalid = False
            if self.is_main_process:
                self.logger.info(
                    "loss is non-finite, cancel accumulated backward/optimizer/scheduler/EMA"
                )
            return
        super().backward_step(loss)
        optimizer_skipped = any(
            bool(getattr(optimizer, "step_was_skipped", False))
            for optimizer in self.optimizers
        )
        self._optimizer_step_succeeded = (
            loss_is_finite and self.accelerator.sync_gradients and not optimizer_skipped
        )
        if self.target_value_ema is not None:
            online = self.accelerator.unwrap_model(
                self.model, keep_torch_compile=False
            )
            self.target_value_ema.update(
                online.value_expert,
                optimizer_step=self.cur_step,
                optimizer_step_succeeded=self._optimizer_step_succeeded,
            )

    def print_step(self) -> None:
        pending_eval = self._pending_pixel_eval
        self._pending_pixel_eval = None
        if self._optimizer_step_succeeded and pending_eval is not None:
            self._run_fixed_horizon_eval(pending_eval)
        if (
            self.target_value_ema is not None
            and self.cur_step % self.log_interval == 0
        ):
            ema_object = self.target_value_ema
            self._accumulate_metric(
                "posttrain/ema_updates",
                torch.tensor(
                    float(ema_object.update_count), device=self.device
                ),
            )
            self._accumulate_metric(
                "posttrain/ema_online_l2",
                torch.tensor(ema_object.last_online_l2, device=self.device),
            )
        self._optimizer_step_succeeded = False
        super().print_step()

    def _accumulate_metric(self, name: str, value: Tensor, *, total: bool = False) -> None:
        scalar = value.detach().float().reshape(())
        gathered = self.accelerator.gather(scalar[None]).reshape(-1)
        reduced = gathered.sum() if total else gathered.mean()
        if name not in self._outputs:
            self._outputs[name] = {"sum": 0.0, "num": 0}
        self._outputs[name]["sum"] += float(reduced.cpu().item())
        self._outputs[name]["num"] += 1

    def _record_posttrain_metrics(self) -> None:
        for name, value in self._posttrain_metrics.items():
            self._accumulate_metric(
                name,
                value,
                total=name.endswith("_samples"),
            )
        self._posttrain_metrics = {}

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
            "pool_id": batch_dict.get(
                "pool_id", torch.zeros_like(batch_dict["sample_index"])
            )[0].detach().cpu(),
            "context": context[:1].detach(),
            "context_mask": context_mask[:1].detach(),
            "current": current[:1].detach(),
            "state": state[:1].detach(),
            "action": action[:1].detach(),
        }

    def _pixel_eval_dataset(self, pool_id: int):
        dataset = self.dataloader.dataset
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        children = getattr(dataset, "datasets", None)
        if children is not None:
            if not 0 <= int(pool_id) < len(children):
                raise IndexError(f"pixel-eval pool_id {pool_id} is outside {len(children)} pools")
            dataset = children[int(pool_id)]
        while not hasattr(dataset, "load_eval_future_latents") and hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        if not hasattr(dataset, "load_eval_future_latents") or not hasattr(dataset, "eval_horizons"):
            raise TypeError("pixel eval requires RoboTwinHDF5Dataset eval accessors")
        return dataset

    def _run_fixed_horizon_eval(self, payload: dict[str, Tensor]) -> None:
        dataset = self._pixel_eval_dataset(int(payload.get("pool_id", torch.tensor(0)).item()))
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
        reward_template = torch.empty(count, 1, 1, device=self.device, dtype=self.dtype)
        q_template = torch.empty(count, 1, 1, device=self.device, dtype=self.dtype)
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
                # Training-time pixel monitoring evaluates only Stage 2 under
                # the batch's full-clean GT action teacher-forcing track.
                future_noise = torch.randn_like(future_template)
                future_state_noise = torch.randn_like(future_state_template)
                reward_query = torch.zeros_like(reward_template)
                q_noise = torch.randn_like(q_template)
                clean_action_time = torch.zeros(count, device=self.device, dtype=torch.float32)

                def predict_world(
                    sampled_future: Tensor,
                    sampled_future_state: Tensor,
                    reward_query_: Tensor,
                    sampled_q: Tensor,
                    sampled_action: Tensor,
                    sigma: Tensor,
                ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
                        noisy_reward=reward_query_,
                        noisy_q=sampled_q,
                        action_timestep=clean_action_time,
                        wm_timestep=wm_time,
                        context_mask=eval_context_mask,
                    )
                    return (
                        world_output.image,
                        world_output.future_state,
                        world_output.reward,
                        world_output.success,
                        world_output.q,
                    )

                samples = sample_world_flow(
                    clean_action=action[:1].expand(count, -1, -1),
                    future_noise=future_noise,
                    future_state_noise=future_state_noise,
                    reward_template=reward_query,
                    q_noise=q_noise,
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
                def to_uint8(images: Tensor) -> Tensor:
                    return images.mul(255).round().to(torch.uint8)

                local_current = to_uint8(local_current)
                local_targets = to_uint8(local_targets)
                local_predictions = to_uint8(local_predictions)
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
        gathered_horizons = self.accelerator.gather(horizons.unsqueeze(0))

        try:
            if self.is_main_process:
                decoded_current = gathered_current.float().div(255).cpu()
                decoded_targets = gathered_targets.float().div(255).cpu()
                decoded_predictions = gathered_predictions.float().div(255).cpu()
                log_pixel_eval(
                    accelerator=self.accelerator,
                    step=self.cur_step,
                    current=decoded_current,
                    targets=decoded_targets,
                    predictions=decoded_predictions,
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


    def _mac_real_batch(self, batch_dict: dict[str, Any]) -> dict[str, Tensor]:
        """Move and validate the fixed-48 fields shared by both MAC phases."""

        context = batch_dict["context"].to(device=self.device, dtype=self.dtype)
        current = batch_dict["current_latents"].to(device=self.device, dtype=self.dtype)
        future = batch_dict["future_latents"].to(device=self.device, dtype=self.dtype)
        state = batch_dict["state"].to(device=self.device, dtype=self.dtype).unsqueeze(1)
        future_state = batch_dict["future_state"].to(
            device=self.device, dtype=self.dtype
        ).unsqueeze(1)
        action = batch_dict.get("behavior_action", batch_dict["action"]).to(
            device=self.device, dtype=self.dtype
        )
        horizon = batch_dict["horizon_idx"].to(
            device=self.device, dtype=torch.long
        ).reshape(-1)
        if not bool(torch.all(horizon == 48)):
            raise ValueError("mac_mot_v2 batches must use the fixed 48-step horizon")
        expected_tokens = self.grid_height * self.grid_width
        if current.shape[1] != expected_tokens or future.shape[1] != expected_tokens:
            raise ValueError(f"cached FLUX image tensors must contain {expected_tokens} tokens")
        return {
            "context": context,
            "context_mask": batch_dict["context_mask"].to(
                device=self.device, dtype=torch.bool
            ),
            "current": current,
            "future": future,
            "state": state,
            "future_state": future_state,
            "action": action,
            "horizon": horizon,
        }

    def _forward_step_mac_world_policy(
        self, batch_dict: dict[str, Any]
    ) -> dict[str, Tensor]:
        """Phase 1: train the single FLUX policy/world model on real data."""

        values = self._mac_real_batch(batch_dict)
        context = values["context"]
        batch = context.shape[0]
        action_timestep = self._sample_timestep(batch)
        world_timestep = self._sample_timestep(batch)
        noisy_action, action_target = flow_noise(values["action"], action_timestep)
        noisy_future, image_target = flow_noise(values["future"], world_timestep)
        noisy_state, state_target = flow_noise(values["future_state"], world_timestep)
        context_ids = text_position_ids(batch, context.shape[1], self.device)
        current_ids = image_position_ids(
            batch,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            time_coord=torch.zeros_like(values["horizon"]),
            device=self.device,
        )
        future_ids = image_position_ids(
            batch,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            time_coord=values["horizon"],
            device=self.device,
        )
        empty = values["action"].new_empty(batch, 0, 1)
        output = self.model(
            context=context,
            context_ids=context_ids,
            current_latents=values["current"],
            current_ids=current_ids,
            noisy_future_latents=noisy_future,
            future_ids=future_ids,
            state=values["state"],
            noisy_pred_action=noisy_action,
            gt_action_cond=values["action"],
            horizon_idx=values["horizon"],
            noisy_future_state=noisy_state,
            noisy_reward=empty,
            noisy_q=empty,
            action_timestep=action_timestep,
            wm_timestep=world_timestep,
            context_mask=values["context_mask"],
        )
        action_mask = batch_dict["action_loss_mask"].to(device=self.device)
        reward_mask = batch_dict["reward_chunk_mask"].to(
            device=self.device, dtype=self.dtype
        )
        losses = {
            "image_loss": masked_mse(output.image, image_target),
            # Dataset sets this mask to success.  Failure trajectories train
            # every world target below but cannot pull the BC policy backward.
            "action_loss": masked_mse(output.action, action_target, action_mask),
            "future_state_loss": masked_mse(output.future_state, state_target),
            "reward_loss": masked_elementwise_bce_with_logits(
                output.reward,
                batch_dict["reward_chunk"].to(device=self.device, dtype=self.dtype),
                reward_mask,
            ),
            "success_loss": masked_bce_with_logits(
                output.success,
                batch_dict["success"].to(device=self.device, dtype=self.dtype).reshape(batch, 1),
            ),
        }
        self._posttrain_metrics.update(
            {
                "posttrain/action_bc_fraction": action_mask.float().mean(),
                "posttrain/reward_valid_fraction": reward_mask.float().mean(),
            }
        )
        return losses

    def _forward_step_mac_critic(
        self, batch_dict: dict[str, Any]
    ) -> dict[str, Tensor]:
        """Phase 2: freeze FLUX and fit online V/Q to one H=1 rollout."""

        if self.target_value_ema is None:
            raise RuntimeError("critic phase requires the target Value expert")
        values = self._mac_real_batch(batch_dict)
        batch = values["context"].shape[0]
        imagination = dict(self.posttrain_config["imagination"])
        candidate_count = int(imagination["candidate_count"])
        schedule = flow_euler_schedule(
            int(imagination["sampling_steps"]),
            flow_shift=float(imagination["flow_shift"]),
            device=self.device,
        )
        # Imagination is stop-gradient and rank-local. Bypass the DDP wrapper
        # for its many Euler forwards so the reducer sees only the final
        # differentiable V/Q forward below.
        rollout_model = self.accelerator.unwrap_model(
            self.model, keep_torch_compile=False
        )
        with evaluating(rollout_model), torch.autocast(
            device_type=self.device.type,
            dtype=self.ema_forward_autocast_dtype,
            enabled=self.device.type == "cuda",
        ):
            imaginary = generate_mac_imaginary_rollout_h1(
                online_model=rollout_model,
                target_value_expert=self.target_value_ema.model,
                context=values["context"],
                current_latents=values["current"],
                state=values["state"],
                context_mask=values["context_mask"],
                candidate_count=candidate_count,
                action_noise=torch.randn(
                    batch,
                    candidate_count,
                    48,
                    values["action"].shape[-1],
                    device=self.device,
                    dtype=self.dtype,
                ),
                future_noise=torch.randn_like(values["future"]),
                future_state_noise=torch.randn_like(values["future_state"]),
                schedule=schedule,
                discount=float(self.posttrain_config["discount"]),
                reward_non_goal=float(self.posttrain_config["reward_non_goal"]),
                reward_goal=float(self.posttrain_config["reward_goal"]),
                return_scale=float(self.posttrain_config["return_scale"]),
                grid_height=self.grid_height,
                grid_width=self.grid_width,
            )
        value_prediction, q_prediction = evaluate_mac_critics(
            model=self.model,
            context=values["context"],
            current_latents=values["current"],
            state=values["state"],
            context_mask=values["context_mask"],
            clean_action=imaginary.selected_action,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )
        scale = float(self.posttrain_config["return_scale"])
        losses = {
            "value_loss": deterministic_return_loss(
                value_prediction, imaginary.value_target_return, return_scale=scale
            ),
            "q_loss": deterministic_return_loss(
                q_prediction, imaginary.q_target_return, return_scale=scale
            ),
        }
        self._posttrain_metrics.update(
            {
                "posttrain/imaginary_chunk_return": imaginary.chunk_return.mean(),
                "posttrain/target_next_value": imaginary.target_next_value.mean(),
                "posttrain/online_next_value": imaginary.online_next_value.mean(),
                "posttrain/value_target_return": imaginary.value_target_return.mean(),
                "posttrain/q_target_return": imaginary.q_target_return.mean(),
                "posttrain/imaginary_success_probability": imaginary.success_logit.float().sigmoid().mean(),
                "posttrain/candidate_q_mean": imaginary.candidate_q.float().mean() * scale,
                "posttrain/candidate_q_std": imaginary.candidate_q.float().std(unbiased=False) * scale,
            }
        )
        return losses

    def forward_step(self, batch_dict: dict[str, Any]):
        if getattr(self, "mac_enabled", False):
            if self.mac_phase == "world_policy":
                return self._forward_step_mac_world_policy(batch_dict)
            return self._forward_step_mac_critic(batch_dict)
        context = batch_dict["context"].to(device=self.device, dtype=self.dtype)
        current = batch_dict["current_latents"].to(device=self.device, dtype=self.dtype)
        future = batch_dict["future_latents"].to(device=self.device, dtype=self.dtype)
        future_dino = None
        if self.dino_dim is not None:
            if self.dino_encoder is None:
                raise RuntimeError("DINO-enabled model is missing its frozen online encoder")
            if "future_dino_images" not in batch_dict:
                raise KeyError(
                    "DINO-enabled training requires future_dino_images from the horizon-selected frame"
                )
            future_dino = self.dino_encoder.encode_views(
                batch_dict["future_dino_images"],
                view_keys=ROBOTWIN_VIEW_KEYS,
                inference_batch_size=self.dino_encoder_batch_size,
            ).to(dtype=self.dtype)
        state = batch_dict["state"].to(device=self.device, dtype=self.dtype).unsqueeze(1)
        behavior_action = batch_dict.get("behavior_action", batch_dict["action"]).to(
            device=self.device, dtype=self.dtype
        )
        future_state = batch_dict["future_state"].to(device=self.device, dtype=self.dtype).unsqueeze(1)
        reward = batch_dict["reward"].to(device=self.device, dtype=self.dtype).reshape(
            context.shape[0], 1, 1
        )
        success = batch_dict["success"].to(device=self.device, dtype=self.dtype).reshape(
            context.shape[0], 1, 1
        )
        accumulated_reward = batch_dict["reward_h"].to(
            device=self.device, dtype=self.dtype
        ).reshape(context.shape[0], 1, 1)
        q = batch_dict["q"].to(device=self.device, dtype=self.dtype).reshape(
            context.shape[0], 1, 1
        )
        horizon = batch_dict["horizon_idx"].to(device=self.device, dtype=torch.long).reshape(-1)
        context_mask = batch_dict["context_mask"].to(device=self.device, dtype=torch.bool)
        action_loss_mask = batch_dict["action_loss_mask"].to(device=self.device)
        q_loss_mask = batch_dict.get("q_loss_mask")
        if q_loss_mask is not None:
            q_loss_mask = q_loss_mask.to(device=self.device)

        pred_action_target = behavior_action

        batch_size = context.shape[0]
        expected_tokens = self.grid_height * self.grid_width
        if current.shape[1] != expected_tokens or future.shape[1] != expected_tokens:
            raise ValueError(
                f"cached FLUX image tokens must use {self.grid_height}x{self.grid_width}={expected_tokens} tokens"
            )
        action_timestep = self._sample_timestep(batch_size)
        wm_timestep = self._sample_timestep(batch_size)
        noisy_action, action_target = flow_noise(pred_action_target, action_timestep)
        noisy_future, image_target = flow_noise(future, wm_timestep)
        noisy_future_state, future_state_target = flow_noise(future_state, wm_timestep)
        reward_query = torch.zeros_like(reward)
        # The direct reward head is a Bernoulli classifier: class 1 means the
        # selected future state is a successful terminal (reward 0), while
        # class 0 means the per-step reward is -1.
        reward_target = success
        noisy_q, q_target = flow_noise(q, wm_timestep)
        noisy_future_dino = None
        dino_target = None
        if future_dino is not None:
            noisy_future_dino, dino_target = flow_noise(future_dino, wm_timestep)

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
        dino_ids = None
        if future_dino is not None:
            if tuple(future_dino.shape[1:]) != (147, self.dino_dim):
                raise ValueError(
                    f"online DINO target must be [B, 147, {self.dino_dim}], "
                    f"got {tuple(future_dino.shape)}"
                )
            dino_ids = dino_position_ids(
                batch_size,
                num_cameras=3,
                grid_height=7,
                grid_width=7,
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
            gt_action_cond=behavior_action,
            horizon_idx=horizon,
            noisy_future_state=noisy_future_state,
            noisy_reward=reward_query,
            noisy_q=noisy_q,
            action_timestep=action_timestep,
            wm_timestep=wm_timestep,
            noisy_future_dino=noisy_future_dino,
            dino_ids=dino_ids,
            context_mask=context_mask,
        )

        if should_log_pixel_eval(self.cur_step, self.pixel_eval_interval):
            self._stage_fixed_horizon_eval(
                batch_dict=batch_dict,
                context=context,
                context_mask=context_mask,
                current=current,
                state=state,
                action=behavior_action,
            )
        losses = joint_flow_loss(
            output,
            image_target=image_target,
            action_target=action_target,
            future_state_target=future_state_target,
            reward_target=reward_target,
            success_target=success,
            q_target=q_target,
            dino_target=dino_target,
            action_loss_mask=action_loss_mask,
            q_loss_mask=q_loss_mask,
        )
        return losses

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
        if self.posttrain_enabled:
            self._record_posttrain_metrics()
        return loss
