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
)
from robonana.models.position_ids import dino_position_ids, image_position_ids, text_position_ids
from robonana.sampling import flow_euler_schedule, sample_world_flow
from robonana.training.losses import joint_flow_loss, masked_mse
from robonana.training.optimizer import build_optimizer_param_groups
from robonana.training.posttraining import (
    FullModelEMA,
    TDTargetResult,
    build_td_targets,
    search_failure_candidates,
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
        self.full_model_ema: FullModelEMA | None = None
        self.ema_forward_autocast_dtype = torch.bfloat16
        self.current_collection_round = int(
            self.posttrain_config.get("current_collection_round", 0)
        )
        self._posttrain_metrics: dict[str, Tensor] = {}
        self._last_candidate_search = None
        self._last_td_target: TDTargetResult | None = None
        self._last_failure_timeout_observation_ids: list[str] = []
        if self.posttrain_enabled:
            self._validate_posttrain_config()

    def _validate_posttrain_config(self) -> None:
        if str(self.kwargs.get("q_target_mode", "")) != "td_posttrain":
            raise ValueError("iterative posttraining requires q_target_mode='td_posttrain'")
        ema = dict(self.posttrain_config.get("ema", {}))
        if ema.get("storage_dtype") != "float32":
            raise ValueError("posttrain EMA storage_dtype must be float32")
        if ema.get("forward_autocast_dtype") != "bfloat16":
            raise ValueError("posttrain EMA forward dtype must be bfloat16")
        failure = dict(self.posttrain_config.get("failure_policy_improvement", {}))
        required = {
            "candidate_policy": "online",
            "candidate_count": 8,
            "candidate_horizon": 48,
            "candidate_selection": "argmax",
            "use_behavior_candidate": False,
            "use_advantage_gate": False,
            "use_confidence_gate": False,
            "use_uncertainty_gate": False,
        }
        for name, expected in required.items():
            if failure.get(name) != expected:
                raise ValueError(f"posttrain failure_policy_improvement.{name} must be {expected!r}")
        td = dict(self.posttrain_config.get("td", {}))
        if td.get("next_action_policy") != "ema" or td.get("target_q_model") != "ema":
            raise ValueError("posttrain TD action and Q targets must both use EMA")
        if td.get("bootstrap_success_terminal") is not False:
            raise ValueError("success terminals must not bootstrap")
        if td.get("bootstrap_failure_timeout") is not True:
            raise ValueError("failure time limits must bootstrap")

    def set_ema_models(self) -> None:
        if not self.posttrain_enabled:
            return super().set_ema_models()
        if self.with_ema:
            raise ValueError("disable FACT parameter-buffer EMA when full posttrain EMA is enabled")
        if len(self.models) != 1:
            raise ValueError("iterative posttraining requires one shared Flux2FACTModel")
        ema = dict(self.posttrain_config["ema"])
        self.full_model_ema = FullModelEMA(
            self.models[0],
            decay=float(ema["decay"]),
            update_every_optimizer_steps=int(ema["update_every_optimizer_steps"]),
            start_step=int(ema["start_step"]),
            device=self.device,
        )

    def prepare(self, dataloaders: Any, models: Any, optimizers: Any, schedulers: Any) -> None:
        super().prepare(dataloaders, models, optimizers, schedulers)
        if self.full_model_ema is not None:
            for optimizer in self.optimizers:
                self.full_model_ema.assert_not_in_optimizer(optimizer)
            if self.is_main_process:
                self.logger.info(
                    "Initialized full FP32 EMA: decay=%.6f updates=%d parameters=%d",
                    self.full_model_ema.decay,
                    self.full_model_ema.update_count,
                    sum(parameter.numel() for parameter in self.full_model_ema.model.parameters()),
                )

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        if self.posttrain_enabled:
            state.update(
                ema_update_count=(
                    0 if self.full_model_ema is None else self.full_model_ema.update_count
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
        if self.full_model_ema is not None:
            self.full_model_ema.update_count = int(state_dict.get("ema_update_count", 0))

    def get_models(self, model_config):
        action_dim = int(_config_value(model_config, "action_dim", 14))
        state_dim = int(_config_value(model_config, "state_dim", 14))
        reward_dim = int(_config_value(model_config, "reward_dim", 1))
        q_dim = int(_config_value(model_config, "q_dim", 1))
        max_horizon = int(_config_value(model_config, "max_horizon", 48))
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
                q_dim=q_dim,
                max_horizon=max_horizon,
                dino_dim=dino_dim,
                pred_action_bidirectional=pred_action_bidirectional,
                device=self.device,
                dtype=self.dtype,
                params=params,
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
                q_dim=q_dim,
                max_horizon=max_horizon,
                dino_dim=dino_dim,
                pred_action_bidirectional=pred_action_bidirectional,
                device=self.device,
                dtype=self.dtype,
                params=params,
                config_path=_config_value(model_config, "checkpoint_config", None),
            )
            initialization_label = (
                f"trained checkpoint parameters={report.checkpoint_parameters}; "
                f"new_parameters={len(report.initialized_robot_parameters)}"
            )
        elif initialization == "scratch":
            model = initialize_flux2_fact_model(
                action_dim=action_dim,
                state_dim=state_dim,
                reward_dim=reward_dim,
                q_dim=q_dim,
                max_horizon=max_horizon,
                dino_dim=dino_dim,
                pred_action_bidirectional=pred_action_bidirectional,
                device=self.device,
                dtype=self.dtype,
                params=params,
            )
            initialization_label = "scratch"
        else:
            raise ValueError(
                "initialization must be 'pretrained', 'trained', or 'scratch', "
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
                "Attention layout: A=%s; G=causal; future_targets=G-prefix-through-idx_h",
                "bidirectional" if model.pred_action_bidirectional else "causal",
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
        if self.full_model_ema is None or not self.is_main_process:
            return
        output = Path(output_dir)
        save_file(
            self.full_model_ema.state_dict(),
            str(output / "ema_model.safetensors"),
        )
        ema_state = {
            "decay": self.full_model_ema.decay,
            "update_every_optimizer_steps": self.full_model_ema.update_every_optimizer_steps,
            "start_step": self.full_model_ema.start_step,
            "update_count": self.full_model_ema.update_count,
            "storage_dtype": "float32",
            "forward_autocast_dtype": "bfloat16",
            "current_collection_round": self.current_collection_round,
        }
        (output / "ema_state.json").write_text(
            json.dumps(ema_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output / "posttrain_config.json").write_text(
            json.dumps(self.posttrain_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def load_model_hook(self, models, input_dir: str) -> None:
        super().load_model_hook(models, input_dir)
        if self.full_model_ema is None:
            return
        ema_path = Path(input_dir) / "ema_model.safetensors"
        state_path = Path(input_dir) / "ema_state.json"
        if ema_path.is_file():
            self.full_model_ema.load_state_dict(load_file(str(ema_path), device="cpu"))
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.full_model_ema.update_count = int(state.get("update_count", 0))
                self.current_collection_round = int(
                    state.get("current_collection_round", self.current_collection_round)
                )
            source = str(ema_path)
        else:
            online = self.accelerator.unwrap_model(
                self.model, keep_torch_compile=False
            )
            self.full_model_ema.exact_copy_from(online)
            self.full_model_ema.update_count = 0
            source = "exact-copy online (legacy checkpoint has no EMA)"
        for optimizer in self.optimizers:
            self.full_model_ema.assert_not_in_optimizer(optimizer)
        if self.is_main_process:
            self.logger.info(
                "Restored full EMA from %s: decay=%.6f updates=%d parameters=%d",
                source,
                self.full_model_ema.decay,
                self.full_model_ema.update_count,
                sum(parameter.numel() for parameter in self.full_model_ema.model.parameters()),
            )

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
        if self.full_model_ema is not None:
            online = self.accelerator.unwrap_model(
                self.model, keep_torch_compile=False
            )
            self.full_model_ema.update(
                online,
                optimizer_step=self.cur_step,
                optimizer_step_succeeded=self._optimizer_step_succeeded,
            )

    def print_step(self) -> None:
        pending_eval = self._pending_pixel_eval
        self._pending_pixel_eval = None
        if self._optimizer_step_succeeded and pending_eval is not None:
            self._run_fixed_horizon_eval(pending_eval)
        if (
            self.full_model_ema is not None
            and self.cur_step % self.log_interval == 0
        ):
            self._accumulate_metric(
                "posttrain/ema_updates",
                torch.tensor(
                    float(self.full_model_ema.update_count), device=self.device
                ),
            )
            self._accumulate_metric(
                "posttrain/ema_online_l2",
                torch.tensor(self.full_model_ema.last_online_l2, device=self.device),
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
                reward_noise = torch.randn_like(reward_template)
                q_noise = torch.randn_like(q_template)
                clean_action_time = torch.zeros(count, device=self.device, dtype=torch.float32)

                def predict_world(
                    sampled_future: Tensor,
                    sampled_future_state: Tensor,
                    sampled_reward: Tensor,
                    sampled_q: Tensor,
                    sampled_action: Tensor,
                    sigma: Tensor,
                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
                        noisy_reward=sampled_reward,
                        noisy_q=sampled_q,
                        action_timestep=clean_action_time,
                        wm_timestep=wm_time,
                        context_mask=eval_context_mask,
                    )
                    return (
                        world_output.image,
                        world_output.future_state,
                        world_output.reward,
                        world_output.q,
                    )

                samples = sample_world_flow(
                    clean_action=action[:1].expand(count, -1, -1),
                    future_noise=future_noise,
                    future_state_noise=future_state_noise,
                    reward_noise=reward_noise,
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

    def _prepare_posttrain_targets(
        self,
        *,
        batch_dict: dict[str, Any],
        context: Tensor,
        context_mask: Tensor,
        current: Tensor,
        future: Tensor,
        state: Tensor,
        future_state: Tensor,
        behavior_action: Tensor,
        reward: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.full_model_ema is None:
            raise RuntimeError("posttraining requires a callable full-model EMA")
        failure_mask = batch_dict["failure_episode_mask"].to(
            device=self.device, dtype=torch.bool
        ).reshape(-1)
        pool_id = batch_dict["pool_id"].to(device=self.device, dtype=torch.long).reshape(-1)
        invalid_failure_pool = failure_mask & ~((pool_id == 2) | (pool_id == 3))
        if bool(invalid_failure_pool.any()):
            raise RuntimeError("failure samples must come from historical/latest failure pools")
        pred_action_target = behavior_action.clone()
        failure = dict(self.posttrain_config["failure_policy_improvement"])
        self._last_candidate_search = None
        if bool(failure_mask.any()):
            result = search_failure_candidates(
                online_model=self.model,
                ema_model=self.full_model_ema.model,
                context=context[failure_mask],
                current_latents=current[failure_mask],
                state=state[failure_mask],
                context_mask=context_mask[failure_mask],
                behavior_action=behavior_action[failure_mask],
                candidate_count=int(failure["candidate_count"]),
                candidate_horizon=int(failure["candidate_horizon"]),
                action_sampling_steps=int(failure["candidate_action_sampling_steps"]),
                q_sampling_steps=int(failure["candidate_q_sampling_steps"]),
                flow_shift=float(failure["flow_shift"]),
                microbatch_size=int(failure["candidate_microbatch_size"]),
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                ema_autocast_dtype=self.ema_forward_autocast_dtype,
            )
            pred_action_target[failure_mask] = result.pseudo_action
            self._last_candidate_search = result
            self._posttrain_metrics.update(
                {
                    "posttrain/failure_best_q_mean": result.best_q.mean(),
                    "posttrain/failure_candidate_q_mean": result.candidate_q.mean(),
                    "posttrain/failure_candidate_q_std": result.candidate_q.std(unbiased=False),
                    "posttrain/failure_behavior_q_mean": result.behavior_q.mean(),
                    "posttrain/failure_best_minus_behavior_q": (
                        result.best_q - result.behavior_q
                    ).mean(),
                    "posttrain/candidate_search_ms": torch.tensor(
                        result.elapsed_ms, device=self.device
                    ),
                    "posttrain/candidate_search_peak_gib": torch.tensor(
                        result.peak_memory_bytes / 1024**3, device=self.device
                    ),
                }
            )

        td = dict(self.posttrain_config["td"])
        td_result = build_td_targets(
            ema_model=self.full_model_ema.model,
            context=context,
            next_current_latents=future,
            next_state=future_state,
            context_mask=context_mask,
            reward_h=reward,
            delta_steps=batch_dict["delta_steps"].to(self.device),
            success_terminal_h=batch_dict["success_terminal_h"].to(self.device),
            action_template=behavior_action,
            discount=float(self.posttrain_config["discount"]),
            target_action_horizon=int(td["target_action_horizon"]),
            action_sampling_steps=int(td["action_sampling_steps"]),
            q_sampling_steps=int(td["q_sampling_steps"]),
            flow_shift=float(td["flow_shift"]),
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            microbatch_size=int(td.get("microbatch_size", 16)),
            ema_autocast_dtype=self.ema_forward_autocast_dtype,
        )
        self._last_td_target = td_result
        delta = batch_dict["delta_steps"].to(device=self.device).float().reshape(-1)
        success_terminal = batch_dict["success_terminal_h"].to(
            device=self.device
        ).float().reshape(-1)
        failure_timeout = batch_dict["time_limit_truncated_h"].to(
            device=self.device
        ).float().reshape(-1)
        valid = td_result.q_loss_mask.reshape(-1) > 0
        bootstrap = td_result.bootstrap_mask.reshape(-1) > 0
        observation_ids = batch_dict.get("observation_id", [])
        if isinstance(observation_ids, (list, tuple)):
            self._last_failure_timeout_observation_ids = [
                str(observation_ids[index])
                for index in range(len(observation_ids))
                if bool(failure_timeout[index] > 0)
            ]
        self._posttrain_metrics.update(
            {
                "posttrain/td_reward_mean": reward.float().mean(),
                "posttrain/td_delta_mean": delta.mean(),
                "posttrain/td_discount_mean": td_result.discount_factor.mean(),
                "posttrain/td_next_q_mean": (
                    td_result.next_q.reshape(-1)[bootstrap].mean()
                    if bool(bootstrap.any())
                    else reward.new_zeros(())
                ),
                "posttrain/td_target_mean": (
                    td_result.q_target.reshape(-1)[valid].mean()
                    if bool(valid.any())
                    else reward.new_zeros(())
                ),
                "posttrain/td_success_terminal_fraction": success_terminal.mean(),
                "posttrain/td_failure_timeout_fraction": failure_timeout.mean(),
                "posttrain/td_bootstrap_fraction": td_result.bootstrap_mask.float().mean(),
                "posttrain/td_target_ms": torch.tensor(td_result.elapsed_ms, device=self.device),
                "posttrain/td_target_peak_gib": torch.tensor(
                    td_result.peak_memory_bytes / 1024**3, device=self.device
                ),
                "posttrain/q_loss_mask_fraction": td_result.q_loss_mask.float().mean(),
                "posttrain/pseudo_action_samples": failure_mask.float().sum(),
                "posttrain/success_action_samples": (~failure_mask).float().sum(),
            }
        )
        pool_metric_names = (
            "original_success_samples",
            "collected_success_samples",
            "historical_failure_samples",
            "latest_failure_samples",
        )
        for index, name in enumerate(pool_metric_names):
            self._posttrain_metrics[f"posttrain/{name}"] = (pool_id == index).float().sum()
        return (
            pred_action_target.detach(),
            td_result.q_target.to(dtype=self.dtype),
            torch.ones_like(failure_mask, dtype=torch.float32),
            td_result.q_loss_mask,
        )

    def forward_step(self, batch_dict: dict[str, Any]):
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
        if self.posttrain_enabled:
            pred_action_target, q, action_loss_mask, q_loss_mask = (
                self._prepare_posttrain_targets(
                    batch_dict=batch_dict,
                    context=context,
                    context_mask=context_mask,
                    current=current,
                    future=future,
                    state=state,
                    future_state=future_state,
                    behavior_action=behavior_action,
                    reward=reward,
                )
            )

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
        noisy_reward, reward_target = flow_noise(reward, wm_timestep)
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
            noisy_reward=noisy_reward,
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
            q_target=q_target,
            dino_target=dino_target,
            action_loss_mask=action_loss_mask,
            q_loss_mask=q_loss_mask,
        )
        if self.posttrain_enabled:
            failure_mask = batch_dict["failure_episode_mask"].to(
                device=self.device, dtype=torch.float32
            )
            self._posttrain_metrics["posttrain/pseudo_action_loss"] = masked_mse(
                output.action,
                action_target,
                failure_mask,
            ).detach()
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
