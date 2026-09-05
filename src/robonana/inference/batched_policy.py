"""True Stage-1 batching for concurrent RoboTwin evaluation requests."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from robonana.encoding import LocalQwen3Embedder, encode_flux2_image_tokens
from robonana.inference.robotwin_policy import (
    InferenceMode,
    RoboNanaRobotWinPolicy,
    postprocess_action,
    seeded_randn_like,
)
from robonana.sampling import QRejectionSample, sample_flux2_action, sample_q_rejection
from world_action_model.image_layouts import (
    ROBOTWIN_VIEW_KEYS,
    build_robotwin_ref_tensor,
)
from world_action_model.pipeline.utils import normalize_state


class BatchedRoboNanaRobotWinPolicy(RoboNanaRobotWinPolicy):
    """Reuse the trained policy while batching independent Stage-1 rollouts.

    Success-rate evaluation only consumes the action chunk. Keeping this fast
    path Stage-1-only avoids paying for auxiliary world heads and VAE decoding
    during every simulator control decision.
    """

    supports_true_batch = True

    def _validate_action_only_batch(self) -> None:
        if self.inference_mode not in {
            InferenceMode.ACTION,
            InferenceMode.ACTION_Q_REJECTION,
        }:
            raise ValueError(
                "batched RoboTwin eval requires action or action_q_rejection"
            )
        if bool(getattr(self, "return_chunk_q", False)) or self.return_stage2_image:
            raise ValueError(
                "batched RoboTwin eval is Stage-1-only; disable auxiliary Stage-2 returns"
            )

    def _batched_context(
        self, observations: Sequence[dict[str, Any]]
    ) -> tuple[Tensor, Tensor]:
        instructions = [
            str(observation.get("instruction", observation.get("prompt", ""))).strip()
            for observation in observations
        ]
        if any(not instruction for instruction in instructions):
            raise ValueError("instruction is empty")
        missing = list(
            dict.fromkeys(
                instruction
                for instruction in instructions
                if instruction not in self._context_cache
            )
        )
        if missing:
            if self._text_embedder is None:
                self._text_embedder = LocalQwen3Embedder(
                    self.flux_checkpoint_dir,
                    self.text_encoder_device,
                )
            encoded = self._text_embedder(missing)
            if encoded.shape[0] != len(missing):
                raise RuntimeError(
                    "Qwen3 context encoder returned a mismatched batch: "
                    f"{encoded.shape[0]} != {len(missing)}"
                )
            for instruction, context in zip(missing, encoded, strict=True):
                self._context_cache[instruction] = context.detach().cpu().contiguous()
        contexts = [
            self._context_cache[instruction].to(
                device=self.model_device,
                dtype=self.dtype,
            )
            for instruction in instructions
        ]
        max_tokens = max(context.shape[0] for context in contexts)
        hidden_dim = contexts[0].shape[-1]
        batch_size = len(contexts)
        padded = torch.zeros(
            batch_size,
            max_tokens,
            hidden_dim,
            device=self.model_device,
            dtype=self.dtype,
        )
        mask = torch.zeros(
            batch_size,
            max_tokens,
            device=self.model_device,
            dtype=torch.bool,
        )
        for index, context in enumerate(contexts):
            length = context.shape[0]
            padded[index, :length] = context
            mask[index, :length] = True
        return padded, mask

    def _batched_current_image_tokens(
        self, observations: Sequence[dict[str, Any]]
    ) -> Tensor:
        composites = []
        for observation in observations:
            images = {
                key: torch.as_tensor(observation[key]) for key in ROBOTWIN_VIEW_KEYS
            }
            composites.append(
                build_robotwin_ref_tensor(images, main_dst_size=self.main_view_size)
            )
        images_nchw = torch.stack(composites, dim=0).to(
            device=self.vae_device,
            dtype=torch.float32,
        )
        tokens = encode_flux2_image_tokens(self.vae, images_nchw.mul(2.0).sub(1.0))
        expected = self.grid_height * self.grid_width
        if tuple(tokens.shape[1:]) != (expected, 128):
            raise RuntimeError(
                f"live FLUX tokens have shape {tuple(tokens.shape)}, "
                f"expected [B, {expected}, 128]"
            )
        return tokens.to(device=self.model_device, dtype=self.dtype)

    @torch.inference_mode()
    def _sample_action_batch(
        self,
        *,
        context: Tensor,
        context_mask: Tensor,
        current: Tensor,
        state: Tensor,
        sampling_seeds: Sequence[int | None],
    ) -> Tensor:
        batch_size = state.shape[0]
        horizon = torch.full(
            (batch_size,),
            self.horizon,
            device=self.model_device,
            dtype=torch.long,
        )
        clean_gt_action = torch.zeros(
            batch_size,
            self.action_chunk,
            self.action_dim,
            device=self.model_device,
            dtype=self.dtype,
        )
        action_noise = torch.cat(
            [
                seeded_randn_like(clean_gt_action[index : index + 1], seed)
                for index, seed in enumerate(sampling_seeds)
            ],
            dim=0,
        )
        if getattr(self, "inference_mode", InferenceMode.ACTION) is InferenceMode.ACTION_Q_REJECTION:
            noise_rows = []
            for batch_index, seed in enumerate(sampling_seeds):
                candidates = []
                for candidate_index in range(self.rejection_candidate_count):
                    candidate_seed = (
                        None
                        if seed is None
                        else int(seed) + 1009 * candidate_index
                    )
                    candidates.append(
                        seeded_randn_like(
                            clean_gt_action[batch_index : batch_index + 1],
                            candidate_seed,
                        )[0]
                    )
                noise_rows.append(torch.stack(candidates, dim=0))
            self._last_batch_rejection = sample_q_rejection(
                model=self.model,
                context=context,
                current_latents=current,
                state=state,
                context_mask=context_mask,
                candidate_count=self.rejection_candidate_count,
                action_noise=torch.stack(noise_rows, dim=0),
                schedule=self.schedule,
                grid_height=self.grid_height,
                grid_width=self.grid_width,
            )
            return self._last_batch_rejection.action
        self._last_batch_rejection = None
        return sample_flux2_action(
            model=self.model,
            context=context,
            current_latents=current,
            state=state,
            context_mask=context_mask,
            action_noise=action_noise,
            horizon_idx=horizon,
            schedule=self.schedule,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )

    @torch.inference_mode()
    def inference_batch(
        self, observations: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Infer one action chunk per observation in a single FLUX flow batch."""

        self._validate_action_only_batch()
        observations = list(observations)
        if not observations:
            raise ValueError("inference_batch requires at least one observation")
        total_start = time.perf_counter()
        batch_size = len(observations)

        raw_states = torch.stack(
            [
                torch.as_tensor(
                    observation["observation.state"],
                    device=self.model_device,
                    dtype=torch.float32,
                ).reshape(-1)[: self.state_dim]
                for observation in observations
            ],
            dim=0,
        )
        if raw_states.shape[-1] != self.state_dim:
            raise ValueError(f"expected state_dim={self.state_dim}, got {raw_states.shape[-1]}")

        start = time.perf_counter()
        current = self._batched_current_image_tokens(observations)
        self._sync(self.vae_device)
        image_encode_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        context, context_mask = self._batched_context(observations)
        language_encode_ms = (time.perf_counter() - start) * 1000.0
        normalized_state = normalize_state(
            raw_states,
            self.normalization,
            mode="zscore",
        ).to(dtype=self.dtype)[:, None]
        sampling_seeds = [
            None
            if observation.get("sampling_seed") is None
            else int(observation["sampling_seed"])
            for observation in observations
        ]

        self._sync(self.model_device)
        start = time.perf_counter()
        sampled_action = self._sample_action_batch(
            context=context,
            context_mask=context_mask,
            current=current,
            state=normalized_state,
            sampling_seeds=sampling_seeds,
        )
        self._sync(self.model_device)
        action_sample_ms = (time.perf_counter() - start) * 1000.0

        actions = [
            postprocess_action(
                sampled_action[index],
                raw_states[index],
                self.normalization,
                delta_mask=self.delta_mask,
            ).cpu()
            for index in range(batch_size)
        ]
        total_ms = (time.perf_counter() - total_start) * 1000.0
        shared_timing = {
            "batch_size": batch_size,
            "image_encode_ms": image_encode_ms,
            "language_encode_ms": language_encode_ms,
            "action_sample_ms": action_sample_ms,
            "total_policy_ms": total_ms,
        }
        responses = [
            {
                "action": action,
                "_inference_mode": self.inference_mode.value,
                "_policy_timing_ms": dict(shared_timing),
                "_sampling_seed": sampling_seeds[index],
            }
            for index, action in enumerate(actions)
        ]
        rejection: QRejectionSample | None = getattr(
            self, "_last_batch_rejection", None
        )
        if rejection is not None:
            for index, response in enumerate(responses):
                candidate_q = (
                    rejection.candidate_q[index].float() * self.q_return_scale
                ).cpu()
                best_index = int(rejection.best_index[index].item())
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
        return responses
