"""A thin shared-backbone extension of the official FLUX.2 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from flux2.model import Flux2, Flux2Params, apply_rope, timestep_embedding

from .attention_mask import (
    MacSegmentMap,
    SegmentMap,
    build_attention_bias,
    build_mac_attention_bias,
)


@dataclass
class Flux2FACTOutput:
    image: Tensor
    action: Tensor
    future_state: Tensor
    reward: Tensor
    success: Tensor
    q: Tensor
    dino: Tensor | None
    segments: SegmentMap | MacSegmentMap
    value: Tensor | None = None


def _masked_attention(q: Tensor, k: Tensor, v: Tensor, bias: Tensor) -> Tensor:
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)
    return rearrange(out, "b h n d -> b n (h d)")


def _expand(value: Tensor, length: int) -> Tensor:
    if value.ndim == 2:
        value = value[:, None, :]
    return value.expand(-1, length, -1)


def _stitch_triple(parts: Iterable[tuple[int, tuple[Tensor, Tensor, Tensor]]]):
    parts = list(parts)
    return tuple(torch.cat([_expand(triple[index], length) for length, triple in parts], dim=1) for index in range(3))


def _stitch_double(parts: Iterable[tuple[int, tuple]]):
    parts = list(parts)
    return (
        _stitch_triple((length, modulation[0]) for length, modulation in parts),
        _stitch_triple((length, modulation[1]) for length, modulation in parts),
    )


class Flux2FACTModel(Flux2):
    """Official FLUX.2 blocks plus minimal robot token adapters and heads."""

    def __init__(
        self,
        params: Flux2Params,
        *,
        action_dim: int,
        state_dim: int,
        reward_dim: int = 1,
        success_dim: int = 1,
        q_dim: int = 1,
        max_horizon: int = 64,
        dino_dim: int | None = None,
        pred_action_bidirectional: bool = False,
    ) -> None:
        super().__init__(params)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.success_dim = success_dim
        self.q_dim = q_dim
        self.max_horizon = max_horizon
        self.dino_dim = None if dino_dim is None else int(dino_dim)
        if self.reward_dim != 1 or self.success_dim != 1 or self.q_dim != 1:
            raise ValueError("reward_dim, success_dim, and q_dim must all be one scalar token")
        if not isinstance(pred_action_bidirectional, bool):
            raise TypeError("pred_action_bidirectional must be a bool")
        self.pred_action_bidirectional = pred_action_bidirectional
        if self.dino_dim is not None and self.dino_dim <= 0:
            raise ValueError("dino_dim must be positive when the DINO branch is enabled")

        self.action_in = nn.Linear(action_dim, self.hidden_size, bias=False)
        self.state_in = nn.Linear(state_dim, self.hidden_size, bias=False)
        # Reward and success are direct predictions, not flow-matched targets.
        # Learned query tokens prevent either target from leaking into the input.
        self.reward_token = nn.Embedding(1, self.hidden_size)
        self.success_token = nn.Embedding(1, self.hidden_size)
        self.q_in = nn.Linear(q_dim, self.hidden_size, bias=False)
        self.horizon_embed = nn.Embedding(max_horizon + 1, self.hidden_size)
        self.segment_embed = nn.Embedding(8, self.hidden_size)
        # Preserve the eight-row embedding used by legacy checkpoints; Q is
        # the one new semantic segment introduced between reward and image.
        self.q_segment_embed = nn.Embedding(1, self.hidden_size)

        self.action_out = nn.Linear(self.hidden_size, action_dim, bias=False)
        self.state_out = nn.Linear(self.hidden_size, state_dim, bias=False)
        self.reward_out = nn.Linear(self.hidden_size, reward_dim, bias=False)
        self.success_out = nn.Linear(self.hidden_size, success_dim, bias=False)
        self.q_out = nn.Linear(self.hidden_size, q_dim, bias=False)
        if self.dino_dim is not None:
            self.dino_in = nn.Linear(self.dino_dim, self.hidden_size)
            self.dino_out = nn.Linear(self.hidden_size, self.dino_dim)
            # Keep the original eight-row segment embedding checkpoint-compatible.
            self.dino_segment_embed = nn.Embedding(1, self.hidden_size)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def _condition_vec(self, timestep: Tensor, guidance: Tensor | None) -> Tensor:
        condition_dtype = self.time_in.in_layer.weight.dtype
        vec = self.time_in(timestep_embedding(timestep, 256).to(dtype=condition_dtype))
        if self.use_guidance_embed:
            if guidance is None:
                raise ValueError("guidance is required by this FLUX.2 configuration")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(dtype=condition_dtype))
        return vec

    @staticmethod
    def _robot_ids(
        *,
        batch_size: int,
        length: int,
        segment_id: int,
        device: torch.device,
        dtype: torch.dtype,
        time_ids: Tensor | None = None,
    ) -> Tensor:
        ids = torch.zeros(batch_size, length, 4, device=device, dtype=dtype)
        ids[..., 0] = segment_id
        if time_ids is None and length > 1:
            time_ids = torch.arange(1, length + 1, device=device, dtype=dtype)[None].expand(batch_size, -1)
        if time_ids is not None:
            ids[..., 1] = time_ids.to(device=device, dtype=dtype)
        return ids

    @staticmethod
    def _double_block_forward(block, img, txt, pe_img, pe_txt, mod_img, mod_txt, bias):
        q, k, v, pe, num_txt, mods = block._prepare_qkv(img, txt, pe_img, pe_txt, mod_img, mod_txt)
        q, k = apply_rope(q, k, pe)
        attn = _masked_attention(q, k, v, bias)
        txt_attn, img_attn = attn[:, :num_txt], attn[:, num_txt:]
        return block._apply_residuals(img, txt, img_attn, txt_attn, mods)

    @staticmethod
    def _single_block_forward(block, hidden, pe, modulation, bias):
        q, k, v, mlp, gate = block._qkv(hidden, modulation)
        q, k = apply_rope(q, k, pe)
        attn = _masked_attention(q, k, v, bias)
        return block._out(hidden, attn, mlp, gate)

    def forward(
        self,
        *,
        context: Tensor,
        context_ids: Tensor,
        current_latents: Tensor,
        current_ids: Tensor,
        noisy_future_latents: Tensor,
        future_ids: Tensor,
        state: Tensor,
        noisy_pred_action: Tensor,
        gt_action_cond: Tensor,
        horizon_idx: Tensor,
        noisy_future_state: Tensor,
        noisy_reward: Tensor,
        noisy_q: Tensor,
        action_timestep: Tensor,
        wm_timestep: Tensor,
        noisy_future_dino: Tensor | None = None,
        dino_ids: Tensor | None = None,
        context_mask: Tensor | None = None,
        guidance: Tensor | None = None,
    ) -> Flux2FACTOutput:
        batch_size = context.shape[0]
        packed_world = horizon_idx.ndim == 2
        if horizon_idx.ndim == 1 and horizon_idx.shape[0] == batch_size:
            horizon_matrix = horizon_idx[:, None]
        elif horizon_idx.ndim == 2 and horizon_idx.shape[0] == batch_size and horizon_idx.shape[1] > 0:
            horizon_matrix = horizon_idx
        else:
            raise ValueError("horizon_idx must have shape [B] or non-empty [B, K]")
        horizon_count = horizon_matrix.shape[1]
        if torch.any(horizon_matrix < 1) or torch.any(horizon_matrix > self.max_horizon):
            raise ValueError(f"horizon_idx must be in [1, {self.max_horizon}]")
        if current_ids.shape[-1] != 4 or context_ids.shape[-1] != 4:
            raise ValueError("FLUX.2 position IDs must have four axes")
        if packed_world:
            if noisy_future_state.shape != (batch_size, horizon_count, self.state_dim):
                raise ValueError(
                    "packed noisy_future_state must have shape "
                    f"{(batch_size, horizon_count, self.state_dim)}"
                )
            if noisy_reward.shape != (batch_size, horizon_count, self.reward_dim):
                raise ValueError(
                    "packed noisy_reward must have shape "
                    f"{(batch_size, horizon_count, self.reward_dim)}"
                )
            if noisy_q.shape != (batch_size, horizon_count, self.q_dim):
                raise ValueError(
                    f"packed noisy_q must have shape {(batch_size, horizon_count, self.q_dim)}"
                )
            if noisy_future_latents.ndim != 4 or noisy_future_latents.shape[:2] != (
                batch_size,
                horizon_count,
            ):
                raise ValueError("packed noisy_future_latents must have shape [B, K, tokens, channels]")
            if future_ids.shape != (*noisy_future_latents.shape[:3], 4):
                raise ValueError("packed future_ids must have shape [B, K, image_tokens, 4]")
            state_blocks = [noisy_future_state[:, index, None] for index in range(horizon_count)]
            reward_blocks = [noisy_reward[:, index, None] for index in range(horizon_count)]
            q_blocks = [noisy_q[:, index, None] for index in range(horizon_count)]
            image_blocks = [noisy_future_latents[:, index] for index in range(horizon_count)]
            image_id_blocks = [future_ids[:, index] for index in range(horizon_count)]
        else:
            if noisy_future_state.ndim != 3 or noisy_future_state.shape[0] != batch_size:
                raise ValueError("noisy_future_state must have shape [B, tokens, state_dim]")
            if noisy_future_state.shape[-1] != self.state_dim:
                raise ValueError(f"future state dimension must be {self.state_dim}")
            if noisy_reward.ndim != 3 or noisy_reward.shape[0] != batch_size:
                raise ValueError("noisy_reward must have shape [B, tokens, reward_dim]")
            if noisy_reward.shape[-1] != self.reward_dim:
                raise ValueError(f"reward dimension must be {self.reward_dim}")
            if noisy_q.ndim != 3 or noisy_q.shape[0] != batch_size:
                raise ValueError("noisy_q must have shape [B, tokens, q_dim]")
            if noisy_q.shape[-1] != self.q_dim:
                raise ValueError(f"Q dimension must be {self.q_dim}")
            if noisy_future_latents.ndim != 3 or noisy_future_latents.shape[0] != batch_size:
                raise ValueError("noisy_future_latents must have shape [B, tokens, channels]")
            if future_ids.shape != (*noisy_future_latents.shape[:2], 4):
                raise ValueError("future_ids must have shape [B, image_tokens, 4]")
            state_blocks = [noisy_future_state]
            reward_blocks = [noisy_reward]
            q_blocks = [noisy_q]
            image_blocks = [noisy_future_latents]
            image_id_blocks = [future_ids]
        if (noisy_future_dino is None) != (dino_ids is None):
            raise ValueError("noisy_future_dino and dino_ids must be provided together")
        if noisy_future_dino is not None:
            if self.dino_dim is None:
                raise ValueError("DINO tokens were provided to a model with dino_dim=None")
            if noisy_future_dino.shape[-1] != self.dino_dim:
                raise ValueError(
                    f"DINO feature dimension must be {self.dino_dim}, got {noisy_future_dino.shape[-1]}"
                )
            if packed_world:
                if noisy_future_dino.ndim != 4 or noisy_future_dino.shape[:2] != (
                    batch_size,
                    horizon_count,
                ):
                    raise ValueError("packed noisy_future_dino must have shape [B, K, tokens, dino_dim]")
                if dino_ids.shape != (*noisy_future_dino.shape[:3], 4):
                    raise ValueError("packed dino_ids must have shape [B, K, DINO tokens, 4]")
                dino_blocks = [noisy_future_dino[:, index] for index in range(horizon_count)]
                dino_id_blocks = [dino_ids[:, index] for index in range(horizon_count)]
            else:
                if noisy_future_dino.ndim != 3 or noisy_future_dino.shape[0] != batch_size:
                    raise ValueError("noisy_future_dino must have shape [B, tokens, dino_dim]")
                if dino_ids.shape != (*noisy_future_dino.shape[:2], 4):
                    raise ValueError("dino_ids must have shape [B, DINO tokens, 4]")
                dino_blocks = [noisy_future_dino]
                dino_id_blocks = [dino_ids]
        else:
            dino_blocks = [None] * horizon_count
            dino_id_blocks = [None] * horizon_count

        dtype = self.img_in.weight.dtype
        device = context.device
        context = context.to(dtype=dtype)
        current_latents = current_latents.to(dtype=dtype)
        noisy_future_latents = noisy_future_latents.to(dtype=dtype)
        state = state.to(dtype=dtype)
        noisy_pred_action = noisy_pred_action.to(dtype=dtype)
        gt_action_cond = gt_action_cond.to(dtype=dtype)
        noisy_future_state = noisy_future_state.to(dtype=dtype)
        noisy_reward = noisy_reward.to(dtype=dtype)
        noisy_q = noisy_q.to(dtype=dtype)
        if noisy_future_dino is not None:
            noisy_future_dino = noisy_future_dino.to(dtype=dtype)
        if packed_world:
            state_blocks = [noisy_future_state[:, index, None] for index in range(horizon_count)]
            reward_blocks = [noisy_reward[:, index, None] for index in range(horizon_count)]
            q_blocks = [noisy_q[:, index, None] for index in range(horizon_count)]
            image_blocks = [noisy_future_latents[:, index] for index in range(horizon_count)]
            if noisy_future_dino is not None:
                dino_blocks = [noisy_future_dino[:, index] for index in range(horizon_count)]
        else:
            state_blocks = [noisy_future_state]
            reward_blocks = [noisy_reward]
            q_blocks = [noisy_q]
            image_blocks = [noisy_future_latents]
            if noisy_future_dino is not None:
                dino_blocks = [noisy_future_dino]

        block_lengths = {
            "horizon": 1,
            "future_state": state_blocks[0].shape[1],
            "reward": reward_blocks[0].shape[1],
            "success": reward_blocks[0].shape[1],
            "q": q_blocks[0].shape[1],
            "future_image": image_blocks[0].shape[1],
            "future_dino": 0 if dino_blocks[0] is None else dino_blocks[0].shape[1],
        }
        for index in range(horizon_count):
            actual = (
                state_blocks[index].shape[1],
                reward_blocks[index].shape[1],
                q_blocks[index].shape[1],
                image_blocks[index].shape[1],
                0 if dino_blocks[index] is None else dino_blocks[index].shape[1],
            )
            expected = (
                block_lengths["future_state"],
                block_lengths["reward"],
                block_lengths["q"],
                block_lengths["future_image"],
                block_lengths["future_dino"],
            )
            if actual != expected:
                raise ValueError("all packed horizon blocks must have identical token counts")
        prefix_lengths = {
            "language": context.shape[1],
            "state": state.shape[1],
            "ref_image": current_latents.shape[1],
            "pred_action": noisy_pred_action.shape[1],
            "gt_action": gt_action_cond.shape[1],
        }
        segments = SegmentMap.from_block_lengths(
            **prefix_lengths,
            block_count=horizon_count,
            **block_lengths,
        )

        txt = self.txt_in(context)
        def with_segment(part: Tensor, segment_id: int) -> Tensor:
            return part + self.segment_embed.weight[segment_id]

        embedded_parts = [
            with_segment(self.state_in(state), 0),
            with_segment(self.img_in(current_latents), 1),
            with_segment(self.action_in(noisy_pred_action), 2),
            with_segment(self.action_in(gt_action_cond), 3),
        ]
        for index in range(horizon_count):
            embedded_parts.extend(
                [
                    with_segment(self.horizon_embed(horizon_matrix[:, index].long())[:, None], 4),
                    with_segment(self.state_in(state_blocks[index]), 5),
                    self.reward_token.weight[None].expand(
                        batch_size, reward_blocks[index].shape[1], -1
                    ),
                    self.success_token.weight[None].expand(
                        batch_size, reward_blocks[index].shape[1], -1
                    ),
                    self.q_in(q_blocks[index]) + self.q_segment_embed.weight[0],
                    with_segment(self.img_in(image_blocks[index]), 7),
                ]
            )
            if dino_blocks[index] is not None:
                embedded_parts.append(
                    self.dino_in(dino_blocks[index]) + self.dino_segment_embed.weight[0]
                )
        img = torch.cat(embedded_parts, dim=1)

        id_dtype = current_ids.dtype
        action_time = torch.arange(1, noisy_pred_action.shape[1] + 1, device=device, dtype=id_dtype)[None]
        action_time = action_time.expand(batch_size, -1)
        gt_time = torch.arange(1, gt_action_cond.shape[1] + 1, device=device, dtype=id_dtype)[None]
        gt_time = gt_time.expand(batch_size, -1)
        nontext_id_parts = [
            self._robot_ids(batch_size=batch_size, length=state.shape[1], segment_id=1, device=device, dtype=id_dtype),
            current_ids.to(device=device),
            self._robot_ids(batch_size=batch_size, length=noisy_pred_action.shape[1], segment_id=3, device=device, dtype=id_dtype, time_ids=action_time),
            self._robot_ids(batch_size=batch_size, length=gt_action_cond.shape[1], segment_id=4, device=device, dtype=id_dtype, time_ids=gt_time),
        ]
        for index in range(horizon_count):
            nontext_id_parts.extend(
                [
                    self._robot_ids(batch_size=batch_size, length=1, segment_id=5, device=device, dtype=id_dtype, time_ids=horizon_matrix[:, index, None]),
                    self._robot_ids(batch_size=batch_size, length=state_blocks[index].shape[1], segment_id=6, device=device, dtype=id_dtype),
                    self._robot_ids(batch_size=batch_size, length=reward_blocks[index].shape[1], segment_id=7, device=device, dtype=id_dtype),
                    self._robot_ids(batch_size=batch_size, length=reward_blocks[index].shape[1], segment_id=9, device=device, dtype=id_dtype),
                    self._robot_ids(batch_size=batch_size, length=q_blocks[index].shape[1], segment_id=8, device=device, dtype=id_dtype),
                    image_id_blocks[index].to(device=device),
                ]
            )
            if dino_id_blocks[index] is not None:
                nontext_id_parts.append(dino_id_blocks[index].to(device=device))
        nontext_ids = torch.cat(nontext_id_parts, dim=1)

        pe_img = self.pe_embedder(nontext_ids)
        pe_txt = self.pe_embedder(context_ids.to(device=device))
        bias = build_attention_bias(
            segments,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            horizon_idx=horizon_matrix,
            pred_action_bidirectional=self.pred_action_bidirectional,
            context_mask=context_mask,
        )

        zeros = torch.zeros_like(wm_timestep)
        vec_clean = self._condition_vec(zeros, guidance)
        vec_action = self._condition_vec(action_timestep, guidance)
        vec_wm = self._condition_vec(wm_timestep, guidance)

        double_clean = self.double_stream_modulation_img(vec_clean)
        double_action = self.double_stream_modulation_img(vec_action)
        double_wm = self.double_stream_modulation_img(vec_wm)
        double_parts = [
            (prefix_lengths["state"], double_clean),
            (prefix_lengths["ref_image"], double_clean),
            (prefix_lengths["pred_action"], double_action),
            (prefix_lengths["gt_action"], double_clean),
        ]
        for _ in range(horizon_count):
            double_parts.extend(
                [
                    (block_lengths["horizon"], double_clean),
                    (block_lengths["future_state"], double_wm),
                    (block_lengths["reward"], double_wm),
                    (block_lengths["success"], double_wm),
                    (block_lengths["q"], double_wm),
                    (block_lengths["future_image"], double_wm),
                    (block_lengths["future_dino"], double_wm),
                ]
            )
        double_img = _stitch_double(double_parts)
        double_txt = self.double_stream_modulation_txt(vec_clean)

        for block in self.double_blocks:
            if self.gradient_checkpointing and self.training:
                img, txt = checkpoint(
                    lambda img_, txt_, block_=block: self._double_block_forward(
                        block_, img_, txt_, pe_img, pe_txt, double_img, double_txt, bias
                    ),
                    img,
                    txt,
                    use_reentrant=False,
                )
            else:
                img, txt = self._double_block_forward(
                    block, img, txt, pe_img, pe_txt, double_img, double_txt, bias
                )

        hidden = torch.cat([txt, img], dim=1)
        pe = torch.cat([pe_txt, pe_img], dim=2)
        single_clean = self.single_stream_modulation(vec_clean)[0]
        single_action = self.single_stream_modulation(vec_action)[0]
        single_wm = self.single_stream_modulation(vec_wm)[0]
        single_parts = [
            (prefix_lengths["language"], single_clean),
            (prefix_lengths["state"], single_clean),
            (prefix_lengths["ref_image"], single_clean),
            (prefix_lengths["pred_action"], single_action),
            (prefix_lengths["gt_action"], single_clean),
        ]
        for _ in range(horizon_count):
            single_parts.extend(
                [
                    (block_lengths["horizon"], single_clean),
                    (block_lengths["future_state"], single_wm),
                    (block_lengths["reward"], single_wm),
                    (block_lengths["success"], single_wm),
                    (block_lengths["q"], single_wm),
                    (block_lengths["future_image"], single_wm),
                    (block_lengths["future_dino"], single_wm),
                ]
            )
        single_mod = _stitch_triple(single_parts)
        for block in self.single_blocks:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    lambda hidden_, block_=block: self._single_block_forward(
                        block_, hidden_, pe, single_mod, bias
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = self._single_block_forward(block, hidden, pe, single_mod, bias)

        action_hidden = hidden[:, segments.pred_action]
        image_hidden = torch.stack(
            [hidden[:, block.future_image] for block in segments.world_blocks], dim=1
        )
        state_hidden = torch.stack(
            [hidden[:, block.future_state] for block in segments.world_blocks], dim=1
        )
        reward_hidden = torch.stack(
            [hidden[:, block.reward] for block in segments.world_blocks], dim=1
        )
        success_hidden = torch.stack(
            [hidden[:, block.success] for block in segments.world_blocks], dim=1
        )
        q_hidden = torch.stack(
            [hidden[:, block.q] for block in segments.world_blocks], dim=1
        )
        dino_hidden = torch.stack(
            [hidden[:, block.future_dino] for block in segments.world_blocks], dim=1
        )
        image_shape = image_hidden.shape
        image_output = self.final_layer(
            image_hidden.reshape(batch_size, -1, self.hidden_size), vec_wm
        ).reshape(*image_shape[:-1], self.img_in.in_features)
        state_output = self.state_out(state_hidden)
        reward_output = self.reward_out(reward_hidden)
        success_output = self.success_out(success_hidden)
        q_output = self.q_out(q_hidden)
        if packed_world:
            state_output = state_output.squeeze(2)
            reward_output = reward_output.squeeze(2)
            success_output = success_output.squeeze(2)
            q_output = q_output.squeeze(2)
        else:
            image_output = image_output[:, 0]
            state_output = state_output[:, 0]
            reward_output = reward_output[:, 0]
            success_output = success_output[:, 0]
            q_output = q_output[:, 0]
            dino_hidden = dino_hidden[:, 0]
        return Flux2FACTOutput(
            image=image_output,
            action=self.action_out(action_hidden),
            future_state=state_output,
            reward=reward_output,
            success=success_output,
            q=q_output,
            dino=self.dino_out(dino_hidden) if self.dino_dim is not None and dino_hidden.shape[-2] else None,
            segments=segments,
        )


class MacFlux2FACTModel(Flux2FACTModel):
    """Fixed-48 MAC schema using the same official FLUX.2 backbone.

    This class intentionally retains the compatible action/state/image/DINO
    module names so a legacy 120k RoboNana checkpoint can warm-start them.  All
    horizon and scalar project heads are replaced by deterministic MAC heads.
    """

    architecture_version = "mac_v1"

    def __init__(
        self,
        params: Flux2Params,
        *,
        action_dim: int,
        state_dim: int,
        chunk_horizon: int = 48,
        reward_dim: int = 48,
        success_dim: int = 1,
        q_dim: int = 1,
        value_dim: int = 1,
        dino_dim: int | None = None,
    ) -> None:
        if int(chunk_horizon) != 48:
            raise ValueError("mac_v1 currently requires chunk_horizon=48")
        if int(reward_dim) != int(chunk_horizon):
            raise ValueError("mac_v1 reward_dim must equal chunk_horizon")
        if (int(success_dim), int(q_dim), int(value_dim)) != (1, 1, 1):
            raise ValueError("mac_v1 success, Q, and Value heads must be scalar")
        super().__init__(
            params,
            action_dim=action_dim,
            state_dim=state_dim,
            reward_dim=1,
            success_dim=1,
            q_dim=1,
            max_horizon=chunk_horizon,
            dino_dim=dino_dim,
            pred_action_bidirectional=True,
        )
        self.chunk_horizon = int(chunk_horizon)
        self.max_horizon = self.chunk_horizon
        self.reward_dim = int(reward_dim)
        self.value_dim = int(value_dim)

        # These modules encode the removed variable-horizon/flow-Q schema and
        # must not appear in a mac_v1 checkpoint.
        del self.q_in
        del self.horizon_embed
        del self.segment_embed
        del self.q_segment_embed
        if hasattr(self, "dino_segment_embed"):
            del self.dino_segment_embed

        self.value_token = nn.Embedding(1, self.hidden_size)
        self.q_token = nn.Embedding(1, self.hidden_size)
        self.mac_segment_embed = nn.Embedding(11, self.hidden_size)
        self.value_out = nn.Linear(self.hidden_size, value_dim, bias=False)
        self.reward_out = nn.Linear(self.hidden_size, reward_dim, bias=False)
        self.q_out = nn.Linear(self.hidden_size, q_dim, bias=False)

    def forward(
        self,
        *,
        context: Tensor,
        context_ids: Tensor,
        current_latents: Tensor,
        current_ids: Tensor,
        noisy_future_latents: Tensor,
        future_ids: Tensor,
        state: Tensor,
        noisy_pred_action: Tensor,
        gt_action_cond: Tensor,
        horizon_idx: Tensor,
        noisy_future_state: Tensor,
        noisy_reward: Tensor,
        noisy_q: Tensor,
        action_timestep: Tensor,
        wm_timestep: Tensor,
        noisy_future_dino: Tensor | None = None,
        dino_ids: Tensor | None = None,
        context_mask: Tensor | None = None,
        guidance: Tensor | None = None,
    ) -> Flux2FACTOutput:
        del horizon_idx, noisy_reward, noisy_q
        batch_size = context.shape[0]
        if current_ids.shape != (*current_latents.shape[:2], 4):
            raise ValueError("current_ids must have shape [B, image_tokens, 4]")
        if future_ids.shape != (*noisy_future_latents.shape[:2], 4):
            raise ValueError("future_ids must have shape [B, image_tokens, 4]")
        if context_ids.shape != (*context.shape[:2], 4):
            raise ValueError("context_ids must have shape [B, text_tokens, 4]")
        for name, value, width in (
            ("state", state, self.state_dim),
            ("noisy_future_state", noisy_future_state, self.state_dim),
            ("noisy_pred_action", noisy_pred_action, self.action_dim),
            ("gt_action_cond", gt_action_cond, self.action_dim),
        ):
            if value.ndim != 3 or value.shape[0] != batch_size or value.shape[-1] != width:
                raise ValueError(f"{name} must have shape [B, tokens, {width}]")
        if noisy_pred_action.shape[1] not in (0, self.chunk_horizon):
            raise ValueError("mac_v1 predicted action must be empty or one 48-step chunk")
        if gt_action_cond.shape[1] not in (0, self.chunk_horizon):
            raise ValueError("mac_v1 clean action must be empty or one 48-step chunk")
        if (noisy_future_dino is None) != (dino_ids is None):
            raise ValueError("noisy_future_dino and dino_ids must be provided together")
        if noisy_future_dino is not None:
            if self.dino_dim is None or noisy_future_dino.shape[-1] != self.dino_dim:
                raise ValueError("DINO token dimension does not match the mac_v1 model")
            if dino_ids.shape != (*noisy_future_dino.shape[:2], 4):
                raise ValueError("dino_ids must have shape [B, DINO tokens, 4]")

        dtype = self.img_in.weight.dtype
        device = context.device
        context = context.to(dtype=dtype)
        current_latents = current_latents.to(dtype=dtype)
        noisy_future_latents = noisy_future_latents.to(dtype=dtype)
        state = state.to(dtype=dtype)
        noisy_pred_action = noisy_pred_action.to(dtype=dtype)
        gt_action_cond = gt_action_cond.to(dtype=dtype)
        noisy_future_state = noisy_future_state.to(dtype=dtype)
        if noisy_future_dino is not None:
            noisy_future_dino = noisy_future_dino.to(dtype=dtype)

        segments = MacSegmentMap.from_lengths(
            language=context.shape[1],
            state=state.shape[1],
            ref_image=current_latents.shape[1],
            value=1,
            pred_action=noisy_pred_action.shape[1],
            clean_action=gt_action_cond.shape[1],
            q=1,
            reward=1,
            success=1,
            future_state=noisy_future_state.shape[1],
            future_image=noisy_future_latents.shape[1],
            future_dino=0 if noisy_future_dino is None else noisy_future_dino.shape[1],
        )

        def with_segment(part: Tensor, segment_id: int) -> Tensor:
            return part + self.mac_segment_embed.weight[segment_id]

        txt = self.txt_in(context)
        image_parts = [
            with_segment(self.state_in(state), 0),
            with_segment(self.img_in(current_latents), 1),
            with_segment(self.value_token.weight[None].expand(batch_size, 1, -1), 2),
            with_segment(self.action_in(noisy_pred_action), 3),
            with_segment(self.action_in(gt_action_cond), 4),
            with_segment(self.q_token.weight[None].expand(batch_size, 1, -1), 5),
            with_segment(self.reward_token.weight[None].expand(batch_size, 1, -1), 6),
            with_segment(self.success_token.weight[None].expand(batch_size, 1, -1), 7),
            with_segment(self.state_in(noisy_future_state), 8),
            with_segment(self.img_in(noisy_future_latents), 9),
        ]
        if noisy_future_dino is not None:
            image_parts.append(with_segment(self.dino_in(noisy_future_dino), 10))
        img = torch.cat(image_parts, dim=1)

        id_dtype = current_ids.dtype
        action_time = torch.arange(
            1, noisy_pred_action.shape[1] + 1, device=device, dtype=id_dtype
        )[None].expand(batch_size, -1)
        clean_action_time = torch.arange(
            1, gt_action_cond.shape[1] + 1, device=device, dtype=id_dtype
        )[None].expand(batch_size, -1)
        robot_id = lambda length, segment_id, time_ids=None: self._robot_ids(
            batch_size=batch_size,
            length=length,
            segment_id=segment_id,
            device=device,
            dtype=id_dtype,
            time_ids=time_ids,
        )
        id_parts = [
            robot_id(state.shape[1], 1),
            current_ids.to(device=device),
            robot_id(1, 2),
            robot_id(noisy_pred_action.shape[1], 3, action_time),
            robot_id(gt_action_cond.shape[1], 4, clean_action_time),
            robot_id(1, 5),
            robot_id(1, 6),
            robot_id(1, 7),
            robot_id(noisy_future_state.shape[1], 8),
            future_ids.to(device=device),
        ]
        if dino_ids is not None:
            id_parts.append(dino_ids.to(device=device))
        nontext_ids = torch.cat(id_parts, dim=1)
        pe_img = self.pe_embedder(nontext_ids)
        pe_txt = self.pe_embedder(context_ids.to(device=device))
        bias = build_mac_attention_bias(
            segments,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            context_mask=context_mask,
        )

        zeros = torch.zeros_like(wm_timestep)
        vec_clean = self._condition_vec(zeros, guidance)
        vec_action = self._condition_vec(action_timestep, guidance)
        vec_wm = self._condition_vec(wm_timestep, guidance)
        double_clean = self.double_stream_modulation_img(vec_clean)
        double_action = self.double_stream_modulation_img(vec_action)
        double_wm = self.double_stream_modulation_img(vec_wm)
        nontext_lengths = (
            state.shape[1],
            current_latents.shape[1],
            1,
            noisy_pred_action.shape[1],
            gt_action_cond.shape[1],
            1,
            1,
            1,
            noisy_future_state.shape[1],
            noisy_future_latents.shape[1],
            0 if noisy_future_dino is None else noisy_future_dino.shape[1],
        )
        double_modes = (
            double_clean,
            double_clean,
            double_clean,
            double_action,
            double_clean,
            double_clean,
            double_clean,
            double_clean,
            double_wm,
            double_wm,
            double_wm,
        )
        double_img = _stitch_double(zip(nontext_lengths, double_modes, strict=True))
        double_txt = self.double_stream_modulation_txt(vec_clean)
        for block in self.double_blocks:
            if self.gradient_checkpointing and self.training:
                img, txt = checkpoint(
                    lambda img_, txt_, block_=block: self._double_block_forward(
                        block_, img_, txt_, pe_img, pe_txt, double_img, double_txt, bias
                    ),
                    img,
                    txt,
                    use_reentrant=False,
                )
            else:
                img, txt = self._double_block_forward(
                    block, img, txt, pe_img, pe_txt, double_img, double_txt, bias
                )

        hidden = torch.cat([txt, img], dim=1)
        pe = torch.cat([pe_txt, pe_img], dim=2)
        single_clean = self.single_stream_modulation(vec_clean)[0]
        single_action = self.single_stream_modulation(vec_action)[0]
        single_wm = self.single_stream_modulation(vec_wm)[0]
        single_lengths = (context.shape[1], *nontext_lengths)
        single_modes = (
            single_clean,
            single_clean,
            single_clean,
            single_clean,
            single_action,
            single_clean,
            single_clean,
            single_clean,
            single_clean,
            single_wm,
            single_wm,
            single_wm,
        )
        single_mod = _stitch_triple(zip(single_lengths, single_modes, strict=True))
        for block in self.single_blocks:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    lambda hidden_, block_=block: self._single_block_forward(
                        block_, hidden_, pe, single_mod, bias
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = self._single_block_forward(block, hidden, pe, single_mod, bias)

        image_hidden = hidden[:, segments.future_image]
        image_output = self.final_layer(image_hidden, vec_wm)
        dino_output = None
        if self.dino_dim is not None and segments.future_dino.stop > segments.future_dino.start:
            dino_output = self.dino_out(hidden[:, segments.future_dino])
        return Flux2FACTOutput(
            image=image_output,
            action=self.action_out(hidden[:, segments.pred_action]),
            future_state=self.state_out(hidden[:, segments.future_state]),
            reward=self.reward_out(hidden[:, segments.reward]).squeeze(1),
            success=self.success_out(hidden[:, segments.success]).squeeze(1),
            q=self.q_out(hidden[:, segments.q]).squeeze(1),
            dino=dino_output,
            segments=segments,
            value=self.value_out(hidden[:, segments.value]).squeeze(1),
        )
