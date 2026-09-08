"""RoboNana action policy behind FACT's existing RoboTwin socket protocol."""

from __future__ import annotations

import hashlib
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from diffusers.models import AutoencoderKLFlux2
from flux2.model import Flux2Params
from torch import Tensor

from robonana.data.robotwin_hdf5 import ALOHA_DELTA_MASK
from robonana.encoding import LocalQwen3Embedder
from robonana.normalization import load_a_stats, require_a_stats_path
from robonana.image_pipeline import encode_robotwin_observations, MAIN_VIEW_SIZE
from robonana.models.pretrained import load_flux2_fact_trained_checkpoint
from robonana.sampling import (
    QRejectionSample,
    flow_euler_schedule,
    sample_flux2_action,
    sample_q_rejection,
)
from robonana.training.visualization import decode_flux2_tokens
from world_action_model.image_layouts import (
    ROBOTWIN_VIEW_KEYS,
)
from world_action_model.pipeline.utils import (
    NormalizationTensors,
    add_state_to_action,
    denormalize_action,
    extract_normalization_tensors,
    normalize_state,
)


class InferenceMode(str, Enum):
    """Live action paths used for the Q-vs-policy evaluation ablation."""

    ACTION_ONLY = "action_only"
    ACTION_Q_REJECTION = "action_q_rejection"


def _parse_inference_mode(value: str | InferenceMode) -> InferenceMode:
    if isinstance(value, InferenceMode):
        return value
    try:
        return InferenceMode(str(value))
    except ValueError as error:
        choices = ", ".join(mode.value for mode in InferenceMode)
        raise ValueError(f"inference_mode must be one of: {choices}") from error


def seeded_randn_like(reference: Tensor, seed: int | None) -> Tensor:
    """Sample without coupling evaluation noise to the server's global RNG."""

    if seed is None:
        return torch.randn_like(reference)
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def observation_digest(observation: dict[str, Any]) -> str:
    """Return a compact digest for reproducibility diagnostics."""

    digest = hashlib.sha256()
    for key in ("observation.state", *ROBOTWIN_VIEW_KEYS):
        value = torch.as_tensor(observation[key]).detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    instruction = observation.get("instruction", observation.get("prompt", ""))
    digest.update(str(instruction).encode("utf-8"))
    return digest.hexdigest()[:16]


def tensor_digest(value: Tensor) -> str:
    digest = hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def observation_component_digests(observation: dict[str, Any]) -> dict[str, str]:
    instruction = observation.get("instruction", observation.get("prompt", ""))
    components = {
        "state": tensor_digest(torch.as_tensor(observation["observation.state"])),
        "high": tensor_digest(torch.as_tensor(observation["observation.images.cam_high"])),
        "left": tensor_digest(torch.as_tensor(observation["observation.images.cam_left_wrist"])),
        "right": tensor_digest(torch.as_tensor(observation["observation.images.cam_right_wrist"])),
        "instruction": hashlib.sha256(str(instruction).encode("utf-8")).hexdigest()[:16],
    }
    return components


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
    # DO NOT ENABLE action-range clipping: Q and the world model condition on
    # the sampled action, not a clipped replacement. Keep finite actions intact.
    # action = torch.maximum(torch.minimum(action, normalization.action_max), normalization.action_min)
    action = add_state_to_action(
        action,
        raw_state.float(),
        action_chunk=int(action.shape[0]),
        mask=delta_mask,
    )
    fallback = raw_state.float().unsqueeze(0).expand(action.shape[0], -1)[..., : action.shape[-1]]
    action = torch.where(torch.isfinite(action), action, fallback)
    # DO NOT ENABLE state-range clipping either: after restoring absolute joint
    # targets this would again change the action selected/scored by the model.
    # action = torch.maximum(torch.minimum(action, normalization.state_max), normalization.state_min)
    return action


class RoboNanaRobotWinPolicy:
    """Encode a live RoboTwin observation and sample one absolute action chunk."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        model_config: str | Path | None = None,
        flux_checkpoint_dir: str | Path,
        stats_path: str | Path,
        model_device: str | torch.device = "cuda:0",
        vae_device: str | torch.device = "cuda:1",
        text_encoder_device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        action_chunk: int | None = None,
        action_dim: int | None = None,
        state_dim: int | None = None,
        horizon: int | None = None,
        max_horizon: int | None = None,
        num_inference_steps: int | None = None,
        flow_shift: float | None = None,
        grid_height: int = 12,
        grid_width: int = 24,
        main_view_width: int = 256,
        main_view_height: int = 192,
        model_params: Flux2Params | None = None,
        inference_mode: str | InferenceMode = InferenceMode.ACTION_Q_REJECTION,
        vae_decode_batch_size: int = 4,
        discount: float | None = None,
        reward_non_goal: float | None = None,
        reward_goal: float | None = None,
        success_threshold: float | None = None,
        rejection_candidate_count: int | None = None,
        q_return_scale: float | None = None,
    ) -> None:
        self.flux_checkpoint_dir = Path(flux_checkpoint_dir).expanduser().resolve()
        require_a_stats_path(stats_path)  # fail before loading FLUX/Qwen/VAE
        from robonana.inference_contract import resolve_online_contract
        self.inference_contract = resolve_online_contract(checkpoint, flux_checkpoint_dir, dict(
            action_chunk=action_chunk, horizon=horizon, num_inference_steps=num_inference_steps,
            flow_shift=flow_shift, discount=discount, reward_non_goal=reward_non_goal,
            reward_goal=reward_goal, success_threshold=success_threshold,
            rejection_candidate_count=rejection_candidate_count, q_return_scale=q_return_scale,
        ), inference_mode=inference_mode)
        settings = self.inference_contract["sampling"]
        action_chunk, horizon = settings["action_chunk"], settings["horizon"]
        num_inference_steps, flow_shift = settings["num_inference_steps"], settings["flow_shift"]
        discount, reward_non_goal, reward_goal = settings["discount"], settings["reward_non_goal"], settings["reward_goal"]
        success_threshold = settings["success_threshold"]
        rejection_candidate_count, q_return_scale = settings["rejection_candidate_count"], settings["q_return_scale"]
        if (grid_height, grid_width) != (12, 24) or (main_view_width, main_view_height) != MAIN_VIEW_SIZE:
            raise ValueError("Checkpoint image contract requires grid 12x24 and main view 256x192")
        print(f"Verified checkpoint inference contract: {settings}", flush=True)
        self.model_device = torch.device(model_device)
        self.vae_device = torch.device(vae_device)
        self.text_encoder_device = torch.device(text_encoder_device)
        if dtype != torch.float32:
            raise ValueError("RoboNana inference supports FP32 only")
        self.dtype = torch.float32
        self.action_chunk = int(action_chunk)
        self.horizon = int(horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.flow_shift = float(flow_shift)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.main_view_size = (int(main_view_width), int(main_view_height))
        self.inference_mode = _parse_inference_mode(inference_mode)
        self.vae_decode_batch_size = int(vae_decode_batch_size)
        self.discount = float(discount)
        self.reward_non_goal = float(reward_non_goal)
        self.reward_goal = float(reward_goal)
        self.success_threshold = float(success_threshold)
        self.rejection_candidate_count = int(rejection_candidate_count)
        self.q_return_scale = float(q_return_scale)
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must lie in (0, 1]")
        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError("success_threshold must lie in [0, 1]")
        if self.rejection_candidate_count <= 0:
            raise ValueError("rejection_candidate_count must be positive")
        if self.q_return_scale <= 0:
            raise ValueError("q_return_scale must be positive")
        if (
            self.action_chunk <= 0
            or self.num_inference_steps <= 0
            or self.vae_decode_batch_size <= 0
        ):
            raise ValueError(
                "action_chunk, num_inference_steps, "
                "and vae_decode_batch_size must be positive"
            )

        self.model, self.load_report = load_flux2_fact_trained_checkpoint(
            checkpoint,
            action_dim=action_dim,
            state_dim=state_dim,
            max_horizon=max_horizon,
            device=self.model_device,
            dtype=self.dtype,
            params=model_params,
            config_path=model_config,
        )
        self.action_dim = int(self.model.action_dim)
        self.state_dim = int(self.model.state_dim)
        self.max_horizon = int(self.model.max_horizon)
        if self.action_chunk != 48 or self.horizon != 48 or self.max_horizon != 48:
            raise ValueError("mac_mot_v2 live inference requires action_chunk=horizon=max_horizon=48")
        self.model.eval().requires_grad_(False)
        if getattr(self.model, "architecture_version", None) != "mac_mot_v2":
            raise ValueError("live inference requires a mac_mot_v2 checkpoint")
        self.vae = AutoencoderKLFlux2.from_pretrained(
            self.flux_checkpoint_dir,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        ).eval()
        self.vae.requires_grad_(False)
        self.vae.to(self.vae_device)

        stats = load_a_stats(stats_path)
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
        self._last_rejection: QRejectionSample | None = None

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
        if self.main_view_size != MAIN_VIEW_SIZE:
            raise ValueError("Unified image pipeline requires main view 256x192")
        tokens = encode_robotwin_observations(self.vae, [observation])
        expected = self.grid_height * self.grid_width
        if tuple(tokens.shape[1:]) != (expected, 128):
            raise RuntimeError(
                f"live FLUX tokens have shape {tuple(tokens.shape)}, expected [1, {expected}, 128]"
            )
        return tokens.to(device=self.model_device, dtype=self.dtype)

    @torch.inference_mode()
    def _sample_action(
        self,
        *,
        context: Tensor,
        current: Tensor,
        state: Tensor,
        sampling_seed: int | None = None,
    ) -> Tensor:
        action_template = torch.zeros(
            context.shape[0],
            self.action_chunk,
            self.action_dim,
            device=self.model_device,
            dtype=self.dtype,
        )
        context_mask = torch.ones(
            context.shape[0],
            context.shape[1],
            device=self.model_device,
            dtype=torch.bool,
        )
        if self.inference_mode is InferenceMode.ACTION_Q_REJECTION:
            noises = []
            for candidate_index in range(self.rejection_candidate_count):
                candidate_seed = (
                    None
                    if sampling_seed is None
                    else int(sampling_seed) + 1009 * candidate_index
                )
                noises.append(
                    seeded_randn_like(action_template, candidate_seed)[:, None]
                )
            self._last_rejection = sample_q_rejection(
                model=self.model,
                context=context,
                current_latents=current,
                state=state,
                context_mask=context_mask,
                candidate_count=self.rejection_candidate_count,
                action_noise=torch.cat(noises, dim=1),
                schedule=self.schedule,
                grid_height=self.grid_height,
                grid_width=self.grid_width,
            )
            return self._last_rejection.action
        if self.inference_mode is InferenceMode.ACTION_ONLY:
            noise = seeded_randn_like(action_template, sampling_seed)
            self._last_rejection = None
            return sample_flux2_action(
                model=self.model,
                context=context,
                current_latents=current,
                state=state,
                context_mask=context_mask,
                action_noise=noise,
                chunk_horizon=self.action_chunk,
                schedule=self.schedule,
                grid_height=self.grid_height,
                grid_width=self.grid_width,
            )
        raise RuntimeError(f"unsupported inference mode: {self.inference_mode}")

    @torch.inference_mode()
    def _decode_stage2_images(self, future_tokens: Tensor) -> Tensor:
        """Decode ``[B,K,N,C]`` FLUX tokens to FACT's ``[B,C,K,H,W]``."""

        if future_tokens.ndim != 4:
            raise ValueError("packed future tokens must have shape [B, K, N, C]")
        batch_size, horizon_count, token_count, channel_count = future_tokens.shape
        expected_tokens = self.grid_height * self.grid_width
        if token_count != expected_tokens:
            raise ValueError(f"future token count must be {expected_tokens}, got {token_count}")
        flat = future_tokens.reshape(batch_size * horizon_count, token_count, channel_count)
        decoded_chunks = []
        for start in range(0, flat.shape[0], self.vae_decode_batch_size):
            decoded_chunks.append(
                decode_flux2_tokens(
                    self.vae,
                    flat[start : start + self.vae_decode_batch_size].to(device=self.vae_device),
                    grid_height=self.grid_height,
                    grid_width=self.grid_width,
                ).cpu()
            )
        decoded = torch.cat(decoded_chunks, dim=0)
        decoded = decoded.reshape(batch_size, horizon_count, *decoded.shape[1:])
        # FACT's RoboTwin client expects decoded video frames in [-1, 1].
        return decoded.mul(2.0).sub(1.0).permute(0, 2, 1, 3, 4).contiguous()

    @torch.inference_mode()
    def _decode_stage2_image(self, future_tokens: Tensor) -> Tensor:
        """Decode the selected chunk for the maintained world-model report."""

        return self._decode_stage2_images(future_tokens[:, None])

    @torch.inference_mode()
    def inference(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._last_rejection = None
        timing: dict[str, float] = {}
        total_start = time.perf_counter()
        log_digest = os.environ.get("ROBONANA_LOG_INFERENCE_DIGEST", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        input_digest = observation_digest(observation) if log_digest else None

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
        sampling_seed = (
            None
            if observation.get("sampling_seed") is None
            else int(observation["sampling_seed"])
        )
        self._sync(self.model_device)
        start = time.perf_counter()
        sampled_action = self._sample_action(
            context=context,
            current=current,
            state=normalized_state,
            sampling_seed=sampling_seed,
        )
        self._sync(self.model_device)
        timing["action_sample_ms"] = (time.perf_counter() - start) * 1000.0
        action = postprocess_action(
            sampled_action[0],
            raw_state[0],
            self.normalization,
            delta_mask=self.delta_mask,
        )

        if log_digest:
            components = observation_component_digests(observation)
            print(
                "[RoboNana inference] "
                f"sampling_seed={observation.get('sampling_seed')} "
                f"input_digest={input_digest} action_digest={tensor_digest(action)} "
                + " ".join(f"{key}_digest={value}" for key, value in components.items()),
                flush=True,
            )
        response = {
            "action": action.cpu(),
            "_inference_mode": self.inference_mode.value,
            "_q_selection": "argmax" if self.inference_mode is InferenceMode.ACTION_Q_REJECTION else None,
            "_policy_timing_ms": timing,
            "_sampling_seed": observation.get("sampling_seed"),
        }
        if self._last_rejection is not None:
            candidate_q = (
                self._last_rejection.candidate_q[0].float() * self.q_return_scale
            ).cpu()
            best_index = int(self._last_rejection.best_index[0].item())
            sorted_q = candidate_q.sort(descending=True).values
            response.update(
                candidate_q=candidate_q,
                selected_candidate_index=best_index,
                selected_q=float(candidate_q[best_index].item()),
                q_margin=(
                    float((sorted_q[0] - sorted_q[1]).item())
                    if sorted_q.numel() > 1
                    else 0.0
                ),
                candidate_count=self.rejection_candidate_count,
            )
        timing["total_policy_ms"] = (time.perf_counter() - total_start) * 1000.0
        return response
