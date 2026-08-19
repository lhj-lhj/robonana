"""RoboNana action policy behind FACT's existing RoboTwin socket protocol."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from diffusers.models import AutoencoderKLFlux2
from flux2.model import Klein4BParams
from torch import Tensor

from robonana.data.robotwin_hdf5 import ALOHA_DELTA_MASK
from robonana.encoding import LocalQwen3Embedder, encode_flux2_image_tokens
from robonana.models.position_ids import image_position_ids, text_position_ids
from robonana.models.pretrained import load_flux2_fact_trained_checkpoint
from robonana.sampling import flow_euler_schedule, sample_action_flow
from world_action_model.image_layouts import (
    ROBOTWIN_VIEW_KEYS,
    build_robotwin_ref_tensor,
)
from world_action_model.pipeline.utils import (
    NormalizationTensors,
    add_state_to_action,
    denormalize_action,
    extract_normalization_tensors,
    load_stats,
    normalize_state,
)


def _clamp_like(value: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    return torch.maximum(torch.minimum(value, upper), lower)


def postprocess_action(
    normalized_action: Tensor,
    raw_state: Tensor,
    normalization: NormalizationTensors,
    *,
    delta_mask: Tensor,
) -> Tensor:
    """Invert training normalization and restore absolute ALOHA joint targets."""

    action = denormalize_action(normalized_action.float(), normalization, mode="zscore")
    action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
    action = _clamp_like(action, normalization.action_min, normalization.action_max)
    action = add_state_to_action(
        action,
        raw_state.float(),
        action_chunk=int(action.shape[0]),
        mask=delta_mask,
    )
    fallback = raw_state.float().unsqueeze(0).expand(action.shape[0], -1)[..., : action.shape[-1]]
    action = torch.where(torch.isfinite(action), action, fallback)
    return _clamp_like(action, normalization.state_min, normalization.state_max)


class RoboNanaRobotWinPolicy:
    """Encode a live RoboTwin observation and sample one absolute action chunk."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        flux_checkpoint_dir: str | Path,
        stats_path: str | Path,
        model_device: str | torch.device = "cuda:0",
        vae_device: str | torch.device = "cuda:1",
        text_encoder_device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        action_chunk: int = 48,
        action_dim: int = 14,
        state_dim: int = 14,
        horizon: int = 24,
        max_horizon: int = 48,
        num_inference_steps: int = 20,
        flow_shift: float = 1.0,
        grid_height: int = 12,
        grid_width: int = 24,
        main_view_width: int = 256,
        main_view_height: int = 192,
    ) -> None:
        self.flux_checkpoint_dir = Path(flux_checkpoint_dir).expanduser().resolve()
        self.model_device = torch.device(model_device)
        self.vae_device = torch.device(vae_device)
        self.text_encoder_device = torch.device(text_encoder_device)
        self.dtype = dtype
        self.action_chunk = int(action_chunk)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.max_horizon = int(max_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.flow_shift = float(flow_shift)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.main_view_size = (int(main_view_width), int(main_view_height))
        if not 1 <= self.horizon <= self.max_horizon:
            raise ValueError("horizon must lie in [1, max_horizon]")
        if self.action_chunk <= 0 or self.num_inference_steps <= 0:
            raise ValueError("action_chunk and num_inference_steps must be positive")

        self.model, self.load_report = load_flux2_fact_trained_checkpoint(
            checkpoint,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            max_horizon=self.max_horizon,
            device=self.model_device,
            dtype=self.dtype,
            params=Klein4BParams(),
        )
        self.model.eval().requires_grad_(False)
        self.vae = AutoencoderKLFlux2.from_pretrained(
            self.flux_checkpoint_dir,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        ).eval()
        self.vae.requires_grad_(False)
        self.vae.to(self.vae_device)

        stats = load_stats(str(Path(stats_path).expanduser().resolve()))
        self.normalization = extract_normalization_tensors(
            stats,
            device=self.model_device,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )
        self.delta_mask = torch.as_tensor(
            ALOHA_DELTA_MASK[: self.action_dim],
            device=self.model_device,
            dtype=torch.bool,
        )
        self.schedule = flow_euler_schedule(
            self.num_inference_steps,
            flow_shift=self.flow_shift,
            device=self.model_device,
        )
        self._text_embedder: LocalQwen3Embedder | None = None
        self._context_cache: dict[str, Tensor] = {}

    def _sync(self, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _context(self, instruction: str) -> Tensor:
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("instruction is empty")
        context = self._context_cache.get(instruction)
        if context is None:
            if self._text_embedder is None:
                self._text_embedder = LocalQwen3Embedder(
                    self.flux_checkpoint_dir,
                    self.text_encoder_device,
                )
            context = self._text_embedder([instruction])[0].detach().cpu().contiguous()
            self._context_cache[instruction] = context
        return context.unsqueeze(0).to(device=self.model_device, dtype=self.dtype)

    def _current_image_tokens(self, observation: dict[str, Any]) -> Tensor:
        images = {
            key: torch.as_tensor(observation[key])
            for key in ROBOTWIN_VIEW_KEYS
        }
        composite = build_robotwin_ref_tensor(
            images,
            main_dst_size=self.main_view_size,
        )
        images_nchw = composite.unsqueeze(0).to(
            device=self.vae_device,
            dtype=torch.float32,
        )
        images_nchw = images_nchw.mul(2.0).sub(1.0)
        tokens = encode_flux2_image_tokens(self.vae, images_nchw)
        expected = self.grid_height * self.grid_width
        if tuple(tokens.shape[1:]) != (expected, 128):
            raise RuntimeError(
                f"live FLUX tokens have shape {tuple(tokens.shape)}, expected [1, {expected}, 128]"
            )
        return tokens.to(device=self.model_device, dtype=self.dtype)

    @torch.inference_mode()
    def _sample_action(self, *, context: Tensor, current: Tensor, state: Tensor) -> Tensor:
        batch_size = 1
        horizon = torch.tensor([self.horizon], device=self.model_device, dtype=torch.long)
        context_ids = text_position_ids(batch_size, context.shape[1], self.model_device)
        current_ids = image_position_ids(
            batch_size,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            time_coord=torch.zeros_like(horizon),
            device=self.model_device,
        )
        empty_ids = torch.zeros(batch_size, 0, 4, device=self.model_device, dtype=torch.long)
        empty_image = torch.zeros(batch_size, 0, 128, device=self.model_device, dtype=self.dtype)
        empty_state = torch.zeros(batch_size, 0, self.state_dim, device=self.model_device, dtype=self.dtype)
        empty_value = torch.zeros(batch_size, 0, 1, device=self.model_device, dtype=self.dtype)
        clean_gt_action = torch.zeros(
            batch_size,
            self.action_chunk,
            self.action_dim,
            device=self.model_device,
            dtype=self.dtype,
        )
        clean_wm_time = torch.zeros(batch_size, device=self.model_device, dtype=torch.float32)
        context_mask = torch.ones(
            batch_size,
            context.shape[1],
            device=self.model_device,
            dtype=torch.bool,
        )
        action_noise = torch.randn_like(clean_gt_action)

        def predict_action(sampled_action: Tensor, sigma: Tensor) -> Tensor:
            output = self.model(
                context=context,
                context_ids=context_ids,
                current_latents=current,
                current_ids=current_ids,
                noisy_future_latents=empty_image,
                future_ids=empty_ids,
                state=state,
                noisy_pred_action=sampled_action,
                gt_action_cond=clean_gt_action,
                horizon_idx=horizon,
                noisy_future_state=empty_state,
                noisy_value=empty_value,
                action_timestep=sigma.expand(batch_size),
                wm_timestep=clean_wm_time,
                context_mask=context_mask,
            )
            return output.action

        return sample_action_flow(
            action_noise=action_noise,
            schedule=self.schedule,
            predict_action=predict_action,
        )

    @torch.inference_mode()
    def inference(self, observation: dict[str, Any]) -> dict[str, Any]:
        timing: dict[str, float] = {}
        total_start = time.perf_counter()

        raw_state = torch.as_tensor(
            observation["observation.state"],
            device=self.model_device,
            dtype=torch.float32,
        ).reshape(1, -1)[..., : self.state_dim]
        if raw_state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state_dim={self.state_dim}, got {raw_state.shape[-1]}")

        start = time.perf_counter()
        current = self._current_image_tokens(observation)
        self._sync(self.vae_device)
        timing["image_encode_ms"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        instruction = observation.get("instruction", observation.get("prompt", ""))
        context = self._context(str(instruction))
        timing["language_encode_ms"] = (time.perf_counter() - start) * 1000.0

        normalized_state = normalize_state(
            raw_state,
            self.normalization,
            mode="zscore",
        ).to(dtype=self.dtype).unsqueeze(1)
        self._sync(self.model_device)
        start = time.perf_counter()
        normalized_action = self._sample_action(
            context=context,
            current=current,
            state=normalized_state,
        )[0]
        self._sync(self.model_device)
        timing["action_sample_ms"] = (time.perf_counter() - start) * 1000.0

        action = postprocess_action(
            normalized_action,
            raw_state[0],
            self.normalization,
            delta_mask=self.delta_mask,
        )
        timing["total_policy_ms"] = (time.perf_counter() - total_start) * 1000.0
        return {
            "action": action.cpu(),
            "_policy_timing_ms": timing,
        }
