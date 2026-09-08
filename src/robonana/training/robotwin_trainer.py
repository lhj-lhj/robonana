"""FACT training-loop adapter for the shared FLUX.2 RoboNana model."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from fact_train import Trainer, build_optimizer
from fact_train.utils import as_list
from flux2.model import Flux2Params

# Imports register the raw HDF5 dataset and sampler with FACT.
from robonana.data import robotwin_hdf5 as _robotwin_hdf5  # noqa: F401
from robonana.models.pretrained import (
    configure_trainable_parameters,
    load_flux2_fact_trained_checkpoint,
)
from robonana.models.position_ids import image_position_ids, text_position_ids
from robonana.sampling import (
    evaluate_mac_critics,
    flow_euler_schedule,
    generate_mac_imaginary_rollout_h1,
)
from robonana.training.checkpointing import full_deepspeed_checkpoint
from robonana.training.continuation import rebase_loaded_scheduler
from robonana.training.losses import (
    deterministic_return_loss,
    masked_action_mse,
    masked_bce_with_logits,
    masked_elementwise_bce_with_logits,
    masked_mse,
)
from robonana.training.optimizer import build_optimizer_param_groups
from robonana.training.posttraining import (
    ValueExpertEMA,
    evaluating,
    fp32_compute_context,
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
        if kwargs.get("mixed_precision") not in (None, "no"):
            raise ValueError("RoboNana training is FP32-only; mixed_precision must be no")
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
        self.grid_height = int(self.kwargs.get("latent_grid_height", 12))
        self.grid_width = int(self.kwargs.get("latent_grid_width", 24))
        self.flow_shift = float(self.kwargs.get("flow_shift", 1.0))
        self.num_inference_steps = int(self.kwargs.get("num_inference_steps", 20))
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self._optimizer_step_succeeded = False
        self.vae_checkpoint_dir: str | None = None
        self.posttrain_config = dict(self.kwargs.get("posttrain", {}))
        if not self.posttrain_config.get("enabled", False):
            raise ValueError("RoboNanaTrainer requires enabled mac_mot_v2 posttraining")
        self.posttrain_q_target_mode = str(
            self.posttrain_config.get(
                "q_target_mode", self.kwargs.get("q_target_mode", "")
            )
        )
        self.mac_phase = str(self.posttrain_config.get("phase", "world_policy"))
        self.target_value_ema: ValueExpertEMA | None = None
        self.current_collection_round = int(
            self.posttrain_config.get("current_collection_round", 0)
        )
        self._posttrain_metrics: dict[str, Tensor] = {}
        self._validate_posttrain_config()

    def _validate_posttrain_config(self) -> None:
        from robonana.inference_contract import sampling_contract
        settings = sampling_contract(self.posttrain_config)
        if self.num_inference_steps != settings["num_inference_steps"] or self.flow_shift != settings["flow_shift"]:
            raise ValueError("train sampling settings must match posttrain.imagination")
        if (self.grid_height, self.grid_width) != (12, 24):
            raise ValueError("Checkpoint image contract requires latent grid 12x24")
        configured_mode = str(self.kwargs.get("q_target_mode", ""))
        if configured_mode != "mac_mot_v2":
            raise ValueError(
                "the maintained RL path is mac_mot_v2"
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
        if ema.get("target") != "value_expert_only":
            raise ValueError("mac_mot_v2 EMA target must be value_expert_only")

    def set_ema_models(self) -> None:
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
        # A fresh critic phase loads the current round's phase-1 checkpoint,
        # which already carries the preceding round's trained *online* Value.
        # Phase 1 changes FLUX while freezing that expert, so start a new
        # Polyak trajectory from an exact online copy instead of carrying the
        # preceding round's lagging target across the representation change.
        # If this critic phase itself is resumed, ``load_model_hook`` below
        # replaces this copy with the target saved by the same critic run.
        self.target_value_ema.exact_copy_from(self.models[0].value_expert)
        self.target_value_ema.update_count = 0

    def prepare(self, dataloaders: Any, models: Any, optimizers: Any, schedulers: Any) -> None:
        super().prepare(dataloaders, models, optimizers, schedulers)
        from robonana.image_pipeline import validate_training_image_contracts
        validate_training_image_contracts(self.dataloader.dataset, self.vae_checkpoint_dir)
        self._image_inputs_certified = True
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
        reward_head_type = str(_config_value(model_config, "reward_head_type", "binary_chunk"))
        max_horizon = int(_config_value(model_config, "max_horizon", 48))
        architecture_version = str(_config_value(model_config, "architecture_version", "mac_mot_v2"))
        if architecture_version != "mac_mot_v2":
            raise ValueError("RoboNana only supports the mac_mot_v2 architecture")
        chunk_horizon = int(_config_value(model_config, "chunk_horizon", max_horizon))
        value_dim = int(_config_value(model_config, "value_dim", 1))
        expert_hidden_dim = _config_value(model_config, "expert_hidden_dim", None)
        expert_hidden_dim = None if expert_hidden_dim is None else int(expert_hidden_dim)
        if reward_head_type != "binary_chunk" or reward_dim != chunk_horizon:
            raise ValueError(
                "mac_mot_v2 requires reward_head_type='binary_chunk' and reward_dim=chunk_horizon"
            )
        raw_dino_dim = _config_value(model_config, "dino_dim", None)
        if raw_dino_dim is not None:
            raise ValueError("mac_mot_v2 does not support DINO targets")
        pred_action_bidirectional = _config_value(
            model_config, "pred_action_bidirectional", False
        )
        if not isinstance(pred_action_bidirectional, bool):
            raise TypeError("models.pred_action_bidirectional must be a bool")
        params_config = _config_value(model_config, "params", None)
        if params_config is None:
            raise ValueError("models.params must record the complete FLUX.2 architecture")
        params = Flux2Params(**dict(params_config))
        checkpoint = _config_value(model_config, "checkpoint", None)
        if checkpoint is None:
            raise ValueError("trained MAC initialization requires models.checkpoint")
        from robonana.inference_contract import build_contract, read_contract, check_contract, CONTRACT_FILE
        self.inference_contract = build_contract(
            self.posttrain_config, str(_config_value(model_config, "checkpoint_dir"))
        )
        # Missing metadata is never silently promoted to a certified checkpoint.
        # Only a deliberate new Stage-1 adaptation may start from old weights;
        # Stage 2 freezes FLUX and cannot certify a changed input representation.
        contract_path = Path(checkpoint).parent / CONTRACT_FILE
        if not contract_path.is_file() and self.mac_phase == "world_policy" and self.kwargs.get("allow_uncertified_pretrain", False):
            self.logger.warning("Explicit uncertified Stage-1 initialization: %s; old weights remain uncertified", checkpoint)
        else:
            check_contract(read_contract(checkpoint), self.inference_contract)
        model, report = load_flux2_fact_trained_checkpoint(
            str(checkpoint), action_dim=action_dim, state_dim=state_dim,
            reward_dim=reward_dim, success_dim=success_dim, q_dim=q_dim,
            reward_head_type=reward_head_type, max_horizon=max_horizon,
            pred_action_bidirectional=pred_action_bidirectional,
            architecture_version=architecture_version, chunk_horizon=chunk_horizon,
            value_dim=value_dim, expert_hidden_dim=expert_hidden_dim,
            device=self.device, dtype=self.dtype, params=params,
            config_path=_config_value(model_config, "checkpoint_config", None),
        )
        initialization_label = f"trained MAC checkpoint parameters={report.checkpoint_parameters}"
        train_mode = str(_config_value(model_config, "train_mode", "full"))
        trainable_names = configure_trainable_parameters(model, train_mode)
        if bool(_config_value(model_config, "gradient_checkpointing", True)):
            model.enable_gradient_checkpointing()
        else:
            model.disable_gradient_checkpointing()
        model.train()
        self.model_name = "transformer"

        self.vae_checkpoint_dir = str(_config_value(model_config, "checkpoint_dir"))
        if self.is_main_process:
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            self.logger.info(
                "Initialized FLUX.2 backbone=%s; parameters=%d; trainable_parameters=%d; "
                "trainable_tensors=%d; gradient_checkpointing=%s",
                initialization_label,
                parameter_count,
                trainable_count,
                len(trainable_names),
                model.gradient_checkpointing,
            )
            self.logger.info(
                "Attention layout: architecture=%s; A=%s; clean_action=%s",
                architecture_version,
                "bidirectional" if model.pred_action_bidirectional else "causal",
                "full-48" if architecture_version == "mac_mot_v2" else "causal-prefix",
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
        if not getattr(self, "_image_inputs_certified", False) or self.cur_step <= 0:
            raise RuntimeError("Cannot publish checkpoint before training input contracts are validated")
        super().save_model_hook(models, weights, output_dir)
        if self.is_main_process:
            from robonana.inference_contract import write_contract
            directory = Path(output_dir) / self.model_name
            exported = list(directory.glob("*.bin")) + list(directory.glob("*.safetensors"))
            if len(exported) != 1:
                raise RuntimeError(f"Expected one exported transformer weight file: {directory}")
            write_contract(exported[0], self.inference_contract, phase=self.mac_phase, step=self.cur_step)
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
        from robonana.inference_contract import read_contract, check_contract
        directory = Path(input_dir) / self.model_name
        exported = list(directory.glob("*.bin")) + list(directory.glob("*.safetensors"))
        if len(exported) != 1:
            raise RuntimeError(f"Expected one exported transformer weight file: {directory}")
        check_contract(read_contract(exported[0]), self.inference_contract)
        super().load_model_hook(models, input_dir)
        if self.target_value_ema is not None:
            target_path = Path(input_dir) / "target_value_expert.safetensors"
            state_path = Path(input_dir) / "value_ema_state.json"
            missing = [
                str(path)
                for path in (target_path, state_path)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "critic resume checkpoint is incomplete; missing Value EMA files: "
                    + ", ".join(missing)
                )
            self.target_value_ema.load_state_dict(
                load_file(str(target_path), device="cpu")
            )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.target_value_ema.update_count = int(state["update_count"])
            self.current_collection_round = int(
                state.get("current_collection_round", self.current_collection_round)
            )
            source = str(target_path)
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

    def resume(self, checkpoint=None) -> None:
        # Prefer this run's latest checkpoint on subsequent restarts. Only the
        # first launch falls back to the explicitly selected source checkpoint.
        checkpoint = checkpoint or self.get_checkpoint() or self.kwargs.get("resume_from")
        super().resume(checkpoint)
        if not self.kwargs.get("rebase_scheduler_on_resume", False):
            return
        if checkpoint is None or not 0 < self.cur_step < self.max_steps:
            raise ValueError("critic continuation requires a restored step below max_steps")
        if self.target_value_ema is None or self.target_value_ema.update_count != self.cur_step:
            raise ValueError("critic continuation requires matching restored Value EMA progress")
        trainable = [name.removeprefix("module.") for name, param in self.models[0].named_parameters()
                     if param.requires_grad]
        if not trainable or not all(name.startswith(("q_expert.", "value_expert.")) for name in trainable):
            raise ValueError("critic continuation must keep FLUX frozen")
        rates = [rebase_loaded_scheduler(scheduler, self.cur_step) for scheduler in self.schedulers]
        if self.is_main_process:
            self.logger.info(
                "CRITIC CONTINUATION VERIFIED: source=%s step=%d max_steps=%d "
                "Value_EMA_updates=%d LR=%s trainable_tensors=%d precision=FP32",
                checkpoint, self.cur_step, self.max_steps,
                self.target_value_ema.update_count, rates, len(trainable),
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
        checkpoint_interval = self.checkpoint_interval
        with full_deepspeed_checkpoint(getattr(self, "accelerator", None)):
            try:
                if self.cur_step in early_steps and self.cur_step % int(checkpoint_interval):
                    self.checkpoint_interval = 1
                super().save_checkpoint_step()
            finally:
                self.checkpoint_interval = checkpoint_interval

    def backward_step(self, loss: Tensor) -> None:
        bad_flag = (~torch.isfinite(loss.detach()).all()).to(
            device=loss.device, dtype=torch.int32
        )
        if int(getattr(self.accelerator, "num_processes", 1)) > 1:
            # Accelerate supports SUM, not MIN. [finite, nonfinite] must abort
            # both ranks, including when the bad value appears mid-accumulation.
            bad_flag = self.accelerator.reduce(bad_flag, reduction="sum")
        if bool(bad_flag.item()):
            self._optimizer_step_succeeded = False
            # Do not continue with an unfinished DDP reducer or a partially
            # accumulated ZeRO optimizer. All ranks fail before backward/step;
            # recover from the last complete checkpoint, never catch-and-retry
            # this batch inside the same training loop.
            raise FloatingPointError(
                "Non-finite loss on at least one rank; aborting all ranks before "
                "backward/optimizer/scheduler/Value-EMA. Resume from a complete checkpoint."
            )
        super().backward_step(loss)
        optimizer_skipped = any(
            bool(getattr(optimizer, "step_was_skipped", False))
            for optimizer in self.optimizers
        )
        self._optimizer_step_succeeded = (
            self.accelerator.sync_gradients and not optimizer_skipped
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
        horizon = batch_dict["chunk_horizon"].to(
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
            chunk_horizon=values["horizon"],
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
            "action_loss": masked_action_mse(
                output.action, action_target,
                batch_dict["action_valid_mask"].to(device=self.device), action_mask,
            ),
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
        with evaluating(rollout_model), fp32_compute_context(rollout_model):
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
        with fp32_compute_context(rollout_model):
            value_prediction, q_prediction = evaluate_mac_critics(
                model=self.model,
                context=values["context"],
                current_latents=values["current"],
                state=values["state"],
                context_mask=values["context_mask"],
                clean_action=imaginary.selected_action,
                # Both phases of this step now use the same FLUX precision.
                # Keep the cache guard for callers outside this trainer.
                condition_cache=imaginary.condition_cache,
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
        if self.mac_phase == "world_policy":
            return self._forward_step_mac_world_policy(batch_dict)
        if self.mac_phase == "critic":
            return self._forward_step_mac_critic(batch_dict)
        raise ValueError("MAC phase must be world_policy or critic")

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
        self._record_posttrain_metrics()
        return loss
