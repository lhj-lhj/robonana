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
from robonana.encoding import LocalQwen3Embedder, encode_flux2_image_tokens
from robonana.models.pretrained import load_flux2_fact_trained_checkpoint
from robonana.sampling import (
    QRejectionSample,
    WorldFlowSample,
    flow_euler_schedule,
    sample_flux2_action,
    sample_flux2_world,
    sample_q_rejection,
)
from robonana.training.visualization import decode_flux2_tokens
from world_action_model.image_layouts import (
    ROBOTWIN_VIEW_KEYS,
    build_robotwin_ref_tensor,
)
from world_action_model.pipeline.utils import (
    NormalizationTensors,
    add_state_to_action,
    denormalize_action,
    denormalize_state,
    extract_normalization_tensors,
    load_stats,
    normalize_state,
)


class InferenceMode(str, Enum):
    """Supported RoboNana inference graphs."""

    ACTION = "action"
    ACTION_Q_REJECTION = "action_q_rejection"
    ACTION_REWARD_Q = "action_reward_q"
    WORLD_ALL = "world_all"
    WORLD_HORIZON = "world_horizon"


def _parse_inference_mode(value: str | InferenceMode) -> InferenceMode:
    if isinstance(value, InferenceMode):
        return value
    try:
        return InferenceMode(str(value))
    except ValueError as error:
        choices = ", ".join(mode.value for mode in InferenceMode)
        raise ValueError(f"inference_mode must be one of: {choices}") from error


def _clamp_like(value: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    return torch.maximum(torch.minimum(value, upper), lower)


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


def discounted_reward_sum(rewards: Tensor, discount: float) -> Tensor:
    """Discount and sum a one-dimensional per-horizon reward curve."""

    rewards = rewards.float().reshape(-1)
    if not 0.0 < float(discount) <= 1.0:
        raise ValueError("discount must lie in (0, 1]")
    powers = torch.arange(rewards.numel(), device=rewards.device, dtype=rewards.dtype)
    return (rewards * float(discount) ** powers).sum()


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


def preprocess_action_chunk(
    absolute_action: Tensor,
    raw_state: Tensor,
    normalization: NormalizationTensors,
    *,
    delta_mask: Tensor,
) -> Tensor:
    """Apply the exact inverse of ``postprocess_action`` for Stage-2 inputs."""

    if absolute_action.ndim != 2:
        raise ValueError("action_chunk must have shape [T, action_dim]")
    action_dim = normalization.action_mean.shape[-1]
    if absolute_action.shape[-1] != action_dim:
        raise ValueError(f"action_chunk must have action_dim={action_dim}")
    if raw_state.numel() < action_dim:
        raise ValueError(f"observation state must contain at least {action_dim} values")
    if not torch.isfinite(absolute_action).all():
        raise ValueError("action_chunk contains non-finite values")
    delta_mask = delta_mask.to(device=absolute_action.device, dtype=torch.bool)
    delta = absolute_action.float().clone()
    delta[:, delta_mask] -= raw_state.float()[None, :action_dim][:, delta_mask]
    return (
        delta - normalization.action_mean.to(device=delta.device)
    ) / normalization.action_std.to(device=delta.device).clamp_min(1e-8)


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
        dtype: torch.dtype = torch.bfloat16,
        action_chunk: int = 48,
        action_dim: int | None = None,
        state_dim: int | None = None,
        horizon: int = 24,
        max_horizon: int | None = None,
        num_inference_steps: int = 20,
        flow_shift: float = 1.0,
        grid_height: int = 12,
        grid_width: int = 24,
        main_view_width: int = 256,
        main_view_height: int = 192,
        model_params: Flux2Params | None = None,
        inference_mode: str | InferenceMode = InferenceMode.ACTION,
        stage2_image_horizon_batch_size: int = 4,
        vae_decode_batch_size: int = 4,
        return_chunk_q: bool = False,
        return_stage2_image: bool = False,
        discount: float = 0.999,
        reward_non_goal: float = -1.0,
        reward_goal: float = 0.0,
        success_threshold: float = 0.5,
        rejection_candidate_count: int = 32,
        q_return_scale: float = 1000.0,
    ) -> None:
        self.flux_checkpoint_dir = Path(flux_checkpoint_dir).expanduser().resolve()
        self.model_device = torch.device(model_device)
        self.vae_device = torch.device(vae_device)
        self.text_encoder_device = torch.device(text_encoder_device)
        self.dtype = dtype
        self.action_chunk = int(action_chunk)
        self.horizon = int(horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.flow_shift = float(flow_shift)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.main_view_size = (int(main_view_width), int(main_view_height))
        self.inference_mode = _parse_inference_mode(inference_mode)
        self.stage2_image_horizon_batch_size = int(stage2_image_horizon_batch_size)
        self.vae_decode_batch_size = int(vae_decode_batch_size)
        self.return_chunk_q = bool(return_chunk_q)
        self.return_stage2_image = bool(return_stage2_image)
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
        if self.return_stage2_image and not self.return_chunk_q:
            raise ValueError("return_stage2_image requires return_chunk_q Stage-2 sampling")
        if self.inference_mode is not InferenceMode.ACTION and (
            self.return_chunk_q or self.return_stage2_image
        ):
            raise ValueError(
                "legacy return_chunk_q/return_stage2_image flags cannot be combined "
                "with an explicit non-action inference_mode"
            )
        if (
            self.action_chunk <= 0
            or self.num_inference_steps <= 0
            or self.stage2_image_horizon_batch_size <= 0
            or self.vae_decode_batch_size <= 0
        ):
            raise ValueError(
                "action_chunk, num_inference_steps, stage2_image_horizon_batch_size, "
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
        if not 1 <= self.horizon <= self.max_horizon:
            raise ValueError("horizon must lie in [1, max_horizon]")
        if self.action_chunk > self.max_horizon:
            raise ValueError("action_chunk cannot exceed the checkpoint's max_horizon")
        self.model.eval().requires_grad_(False)
        if (
            self.inference_mode is InferenceMode.ACTION_Q_REJECTION
            and getattr(self.model, "architecture_version", None) != "mac_v1"
        ):
            raise ValueError("action_q_rejection requires a mac_v1 checkpoint")
        if (
            getattr(self.model, "architecture_version", None) == "mac_v1"
            and self.inference_mode
            not in {InferenceMode.ACTION, InferenceMode.ACTION_Q_REJECTION}
        ):
            raise ValueError(
                "mac_v1 live inference currently supports action or action_q_rejection"
            )
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
        self._last_rejection = None
        return sample_flux2_action(
            model=self.model,
            context=context,
            current_latents=current,
            state=state,
            context_mask=context_mask,
            action_noise=seeded_randn_like(action_template, sampling_seed),
            horizon_idx=self.horizon,
            schedule=self.schedule,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )

    @torch.inference_mode()
    def _sample_stage2_chunk(
        self,
        *,
        context: Tensor,
        current: Tensor,
        state: Tensor,
        clean_action: Tensor,
        horizons: Tensor,
        include_image: bool,
        sampling_seed: int | None = None,
    ) -> WorldFlowSample:
        """Jointly sample isolated horizon blocks under one clean action track."""

        batch_size = 1
        horizons = torch.as_tensor(
            horizons,
            device=self.model_device,
            dtype=torch.long,
        ).reshape(-1)
        if horizons.numel() == 0:
            raise ValueError("Stage-2 requires at least one horizon")
        if torch.any(horizons < 1) or torch.any(horizons > self.max_horizon):
            raise ValueError(f"Stage-2 horizons must lie in [1, {self.max_horizon}]")
        horizon_count = int(horizons.numel())
        horizon_matrix = horizons[None]
        if include_image:
            image_tokens = self.grid_height * self.grid_width
        else:
            image_tokens = 0
        context_mask = torch.ones(
            batch_size,
            context.shape[1],
            device=self.model_device,
            dtype=torch.bool,
        )
        future_template = torch.zeros(
            batch_size,
            horizon_count,
            image_tokens,
            current.shape[-1],
            device=self.model_device,
            dtype=self.dtype,
        )
        future_state_template = torch.zeros(
            batch_size,
            horizon_count,
            self.state_dim,
            device=self.model_device,
            dtype=self.dtype,
        )
        reward_template = torch.zeros(
            batch_size,
            horizon_count,
            self.model.reward_dim,
            device=self.model_device,
            dtype=self.dtype,
        )
        q_template = torch.zeros(
            batch_size,
            horizon_count,
            self.model.q_dim,
            device=self.model_device,
            dtype=self.dtype,
        )

        def stream_seed(offset: int) -> int | None:
            return None if sampling_seed is None else int(sampling_seed) + int(offset)

        future_noise = seeded_randn_like(future_template, stream_seed(1))
        future_state_noise = seeded_randn_like(future_state_template, stream_seed(2))
        reward_query = reward_template
        q_noise = seeded_randn_like(q_template, stream_seed(4))
        return sample_flux2_world(
            model=self.model,
            context=context,
            current_latents=current,
            state=state,
            context_mask=context_mask,
            clean_action=clean_action,
            horizon_idx=horizon_matrix,
            future_noise=future_noise,
            future_state_noise=future_state_noise,
            reward_template=reward_query,
            q_noise=q_noise,
            schedule=self.schedule,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )

    @torch.inference_mode()
    def _sample_stage2(
        self,
        *,
        context: Tensor,
        current: Tensor,
        state: Tensor,
        clean_action: Tensor,
        horizons: Tensor,
        include_image: bool,
        sampling_seed: int | None = None,
    ) -> WorldFlowSample:
        """Sample all requested horizons, chunking only the dense image suffix."""

        horizons = torch.as_tensor(horizons, device=self.model_device, dtype=torch.long).reshape(-1)
        if horizons.numel() == 0:
            raise ValueError("Stage-2 requires at least one horizon")
        chunk_size = (
            min(self.stage2_image_horizon_batch_size, int(horizons.numel()))
            if include_image
            else int(horizons.numel())
        )
        chunks = []
        for start in range(0, int(horizons.numel()), chunk_size):
            chunk_horizons = horizons[start : start + chunk_size]
            chunk_seed = (
                None
                if sampling_seed is None
                else int(sampling_seed) + 10_000 * int(chunk_horizons[0].item())
            )
            chunks.append(
                self._sample_stage2_chunk(
                    context=context,
                    current=current,
                    state=state,
                    clean_action=clean_action,
                    horizons=chunk_horizons,
                    include_image=include_image,
                    sampling_seed=chunk_seed,
                )
            )
        return WorldFlowSample(
            future=torch.cat([chunk.future for chunk in chunks], dim=1),
            future_state=torch.cat([chunk.future_state for chunk in chunks], dim=1),
            reward=torch.cat([chunk.reward for chunk in chunks], dim=1),
            success=torch.cat([chunk.success for chunk in chunks], dim=1),
            q=torch.cat([chunk.q for chunk in chunks], dim=1),
        )

    @torch.inference_mode()
    def _sample_world(
        self,
        *,
        context: Tensor,
        current: Tensor,
        state: Tensor,
        clean_action: Tensor,
        sampling_seed: int | None = None,
    ) -> WorldFlowSample:
        """Compatibility wrapper for the former one-horizon Stage-2 path."""

        packed = self._sample_stage2(
            context=context,
            current=current,
            state=state,
            clean_action=clean_action,
            horizons=torch.tensor([self.horizon]),
            include_image=True,
            sampling_seed=sampling_seed,
        )
        return WorldFlowSample(
            future=packed.future[:, 0],
            future_state=packed.future_state[:, 0, None],
            reward=packed.reward[:, 0, None],
            success=packed.success[:, 0, None],
            q=packed.q[:, 0, None],
        )

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
        """Decode one final Stage-2 prediction for the legacy response shape."""

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
        needs_input_action = self.inference_mode in {
            InferenceMode.WORLD_ALL,
            InferenceMode.WORLD_HORIZON,
        }
        if needs_input_action:
            if "action_chunk" not in observation:
                raise KeyError(f"{self.inference_mode.value} requires observation['action_chunk']")
            action = torch.as_tensor(
                observation["action_chunk"],
                device=self.model_device,
                dtype=torch.float32,
            )
            expected_action_shape = (self.action_chunk, self.action_dim)
            if tuple(action.shape) != expected_action_shape:
                raise ValueError(
                    f"action_chunk must have shape {expected_action_shape}, got {tuple(action.shape)}"
                )
            sampled_action = preprocess_action_chunk(
                action,
                raw_state[0],
                self.normalization,
                delta_mask=self.delta_mask,
            ).to(dtype=self.dtype)[None]
        else:
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

        world_sample: WorldFlowSample | None = None
        horizons: Tensor | None = None
        include_image = False
        conditional_reward_curve: Tensor | None = None
        conditional_accumulated_reward: float | None = None
        conditional_terminal_horizon: int | None = None
        reward_curve_evaluated = False
        if self.inference_mode is InferenceMode.ACTION_REWARD_Q:
            start = time.perf_counter()
            # A terminal earlier than h=48 necessarily makes the clipped h=48
            # state terminal too.  Query that endpoint first and only pay for
            # the dense 1..48 reward/success curve when it can contain a goal.
            terminal_check_horizons = torch.tensor(
                [self.action_chunk], device=self.model_device, dtype=torch.long
            )
            terminal_check = self._sample_stage2(
                context=context,
                current=current,
                state=normalized_state,
                clean_action=sampled_action,
                horizons=terminal_check_horizons,
                include_image=False,
                sampling_seed=sampling_seed,
            )
            terminal_probability = float(
                terminal_check.success[0, 0].float().sigmoid().item()
            )
            if terminal_probability >= self.success_threshold:
                prefix_horizons = torch.arange(
                    1, self.action_chunk, device=self.model_device, dtype=torch.long
                )
                if prefix_horizons.numel():
                    prefix = self._sample_stage2(
                        context=context,
                        current=current,
                        state=normalized_state,
                        clean_action=sampled_action,
                        horizons=prefix_horizons,
                        include_image=False,
                        sampling_seed=sampling_seed,
                    )
                    world_sample = WorldFlowSample(
                        future=torch.cat([prefix.future, terminal_check.future], dim=1),
                        future_state=torch.cat(
                            [prefix.future_state, terminal_check.future_state], dim=1
                        ),
                        reward=torch.cat([prefix.reward, terminal_check.reward], dim=1),
                        success=torch.cat(
                            [prefix.success, terminal_check.success], dim=1
                        ),
                        q=torch.cat([prefix.q, terminal_check.q], dim=1),
                    )
                else:
                    world_sample = terminal_check
                horizons = torch.arange(
                    1, self.action_chunk + 1, device=self.model_device, dtype=torch.long
                )
                reward_curve_evaluated = True
                probabilities = world_sample.success[0].float().reshape(-1).sigmoid()
                terminal_indices = torch.where(probabilities >= self.success_threshold)[0]
                conditional_terminal_horizon = (
                    int(terminal_indices[0].item()) + 1
                    if terminal_indices.numel()
                    else self.action_chunk
                )
                reward_probabilities = world_sample.reward[0].float().reshape(-1).sigmoid()
                conditional_reward_curve = torch.where(
                    reward_probabilities >= self.success_threshold,
                    torch.full_like(reward_probabilities, self.reward_goal),
                    torch.full_like(reward_probabilities, self.reward_non_goal),
                )
                conditional_accumulated_reward = float(
                    discounted_reward_sum(
                        conditional_reward_curve[:conditional_terminal_horizon],
                        self.discount,
                    ).item()
                )
            else:
                world_sample = terminal_check
                horizons = terminal_check_horizons
                conditional_reward_curve = torch.full(
                    (self.action_chunk,),
                    self.reward_non_goal,
                    device=self.model_device,
                    dtype=torch.float32,
                )
                conditional_accumulated_reward = float(
                    discounted_reward_sum(
                        conditional_reward_curve,
                        self.discount,
                    ).item()
                )
            self._sync(self.model_device)
            timing["stage2_sample_ms"] = (time.perf_counter() - start) * 1000.0
        elif self.inference_mode is InferenceMode.WORLD_ALL:
            horizons = torch.arange(1, self.action_chunk + 1, device=self.model_device)
            # Offline return annotation needs the same packed h=1..T world
            # query without paying for dense FLUX image tokens or VAE decode.
            # Keep the public WORLD_ALL default unchanged for existing callers.
            include_image = bool(observation.get("include_image", True))
        elif self.inference_mode is InferenceMode.WORLD_HORIZON:
            if "horizon" not in observation:
                raise KeyError("world_horizon requires observation['horizon']")
            requested_horizon = torch.as_tensor(observation["horizon"])
            if requested_horizon.numel() != 1:
                raise ValueError("horizon must be one integer scalar")
            horizon_value = int(requested_horizon.item())
            if float(requested_horizon.item()) != float(horizon_value):
                raise ValueError("horizon must be an integer")
            if not 1 <= horizon_value <= self.action_chunk:
                raise ValueError(f"horizon must lie in [1, {self.action_chunk}]")
            horizons = torch.tensor([horizon_value], device=self.model_device)
            include_image = True

        if horizons is not None and world_sample is None:
            start = time.perf_counter()
            world_sample = self._sample_stage2(
                context=context,
                current=current,
                state=normalized_state,
                clean_action=sampled_action,
                horizons=horizons,
                include_image=include_image,
                sampling_seed=sampling_seed,
            )
            self._sync(self.model_device)
            timing["stage2_sample_ms"] = (time.perf_counter() - start) * 1000.0

        # Preserve the former single-horizon live-eval path for old launch
        # scripts. New callers should select one of the four explicit modes.
        legacy_world: WorldFlowSample | None = None
        legacy_stage2_image: Tensor | None = None
        if self.return_chunk_q:
            start = time.perf_counter()
            legacy_world = self._sample_world(
                context=context,
                current=current,
                state=normalized_state,
                clean_action=sampled_action,
                sampling_seed=sampling_seed,
            )
            self._sync(self.model_device)
            timing["legacy_stage2_sample_ms"] = (time.perf_counter() - start) * 1000.0
            if self.return_stage2_image:
                start = time.perf_counter()
                legacy_stage2_image = self._decode_stage2_image(legacy_world.future)
                self._sync(self.vae_device)
                timing["stage2_image_decode_ms"] = (time.perf_counter() - start) * 1000.0

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
        if world_sample is not None and horizons is not None:
            future_states = denormalize_state(
                world_sample.future_state[0].float(), self.normalization, mode="zscore"
            ).cpu()
            reward_probs = world_sample.reward[0].float().reshape(-1).sigmoid().cpu()
            rewards = torch.where(
                reward_probs >= self.success_threshold,
                torch.full_like(reward_probs, self.reward_goal),
                torch.full_like(reward_probs, self.reward_non_goal),
            )
            success_probs = world_sample.success[0].float().reshape(-1).sigmoid().cpu()
            qs = world_sample.q[0].float().reshape(-1).cpu()
            response.update(
                horizons=horizons.detach().cpu(),
                future_states=future_states,
                rewards=rewards,
                reward_probs=reward_probs,
                success_probs=success_probs,
                qs=qs,
            )
            if self.horizon in horizons.tolist():
                selected_index = horizons.tolist().index(self.horizon)
            else:
                selected_index = 0
            response.update(
                chunk_reward=float(rewards[selected_index].item()),
                chunk_q=float(qs[selected_index].item()),
                return_horizon=int(horizons[selected_index].item()),
                selected_index=selected_index,
            )
            if conditional_reward_curve is not None:
                response.update(
                    reward_curve=conditional_reward_curve.detach().cpu(),
                    reward_curve_horizons=torch.arange(1, self.action_chunk + 1),
                    accumulated_reward=conditional_accumulated_reward,
                    terminal_horizon=conditional_terminal_horizon,
                    reward_curve_evaluated=reward_curve_evaluated,
                )
            if include_image:
                response["future_latents"] = world_sample.future[0].detach().cpu()
                start = time.perf_counter()
                response["images"] = self._decode_stage2_images(world_sample.future)
                self._sync(self.vae_device)
                timing["stage2_image_decode_ms"] = (time.perf_counter() - start) * 1000.0
        if legacy_world is not None:
            reward_probability = float(
                legacy_world.reward[0, 0].float().reshape(-1)[0].sigmoid().item()
            )
            chunk_reward = (
                self.reward_goal
                if reward_probability >= self.success_threshold
                else self.reward_non_goal
            )
            chunk_q = float(legacy_world.q[0, 0].float().reshape(-1)[0].item())
            success_probability = float(
                legacy_world.success[0, 0].float().reshape(-1)[0].sigmoid().item()
            )
            response.update(
                chunk_reward=chunk_reward,
                chunk_q=chunk_q,
                return_horizon=self.horizon,
                rewards=torch.tensor([chunk_reward], dtype=torch.float32),
                reward_probs=torch.tensor([reward_probability], dtype=torch.float32),
                qs=torch.tensor([chunk_q], dtype=torch.float32),
                success_probs=torch.tensor([success_probability], dtype=torch.float32),
                selected_index=0,
            )
        if legacy_stage2_image is not None:
            response["images"] = legacy_stage2_image
        timing["total_policy_ms"] = (time.perf_counter() - total_start) * 1000.0
        return response
