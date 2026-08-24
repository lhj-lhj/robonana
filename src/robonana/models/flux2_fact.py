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

from .attention_mask import SegmentMap, build_attention_bias


@dataclass
class Flux2FACTOutput:
    image: Tensor
    action: Tensor
    future_state: Tensor
    value: Tensor
    dino: Tensor | None
    segments: SegmentMap


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
        value_dim: int = 1,
        max_horizon: int = 64,
        dino_dim: int | None = None,
    ) -> None:
        super().__init__(params)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.value_dim = value_dim
        self.max_horizon = max_horizon
        self.dino_dim = None if dino_dim is None else int(dino_dim)
        if self.dino_dim is not None and self.dino_dim <= 0:
            raise ValueError("dino_dim must be positive when the DINO branch is enabled")

        self.action_in = nn.Linear(action_dim, self.hidden_size, bias=False)
        self.state_in = nn.Linear(state_dim, self.hidden_size, bias=False)
        self.value_in = nn.Linear(value_dim, self.hidden_size, bias=False)
        self.horizon_embed = nn.Embedding(max_horizon + 1, self.hidden_size)
        self.segment_embed = nn.Embedding(8, self.hidden_size)

        self.action_out = nn.Linear(self.hidden_size, action_dim, bias=False)
        self.state_out = nn.Linear(self.hidden_size, state_dim, bias=False)
        self.value_out = nn.Linear(self.hidden_size, value_dim, bias=False)
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
        noisy_value: Tensor,
        action_timestep: Tensor,
        wm_timestep: Tensor,
        noisy_future_dino: Tensor | None = None,
        dino_ids: Tensor | None = None,
        context_mask: Tensor | None = None,
        guidance: Tensor | None = None,
    ) -> Flux2FACTOutput:
        batch_size = context.shape[0]
        if horizon_idx.ndim != 1 or horizon_idx.shape[0] != batch_size:
            raise ValueError("horizon_idx must have shape [B]")
        if torch.any(horizon_idx < 1) or torch.any(horizon_idx > self.max_horizon):
            raise ValueError(f"horizon_idx must be in [1, {self.max_horizon}]")
        if current_ids.shape[-1] != 4 or future_ids.shape[-1] != 4 or context_ids.shape[-1] != 4:
            raise ValueError("FLUX.2 position IDs must have four axes")
        if (noisy_future_dino is None) != (dino_ids is None):
            raise ValueError("noisy_future_dino and dino_ids must be provided together")
        if noisy_future_dino is not None:
            if self.dino_dim is None:
                raise ValueError("DINO tokens were provided to a model with dino_dim=None")
            if noisy_future_dino.ndim != 3 or noisy_future_dino.shape[0] != batch_size:
                raise ValueError("noisy_future_dino must have shape [B, tokens, dino_dim]")
            if noisy_future_dino.shape[-1] != self.dino_dim:
                raise ValueError(
                    f"DINO feature dimension must be {self.dino_dim}, got {noisy_future_dino.shape[-1]}"
                )
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
        noisy_value = noisy_value.to(dtype=dtype)
        if noisy_future_dino is not None:
            noisy_future_dino = noisy_future_dino.to(dtype=dtype)

        lengths = {
            "language": context.shape[1],
            "state": state.shape[1],
            "ref_image": current_latents.shape[1],
            "pred_action": noisy_pred_action.shape[1],
            "gt_action": gt_action_cond.shape[1],
            "horizon": 1,
            "future_state": noisy_future_state.shape[1],
            "value": noisy_value.shape[1],
            "future_image": noisy_future_latents.shape[1],
            "future_dino": 0 if noisy_future_dino is None else noisy_future_dino.shape[1],
        }
        segments = SegmentMap.from_lengths(**lengths)

        txt = self.txt_in(context)
        parts = [
            self.state_in(state),
            self.img_in(current_latents),
            self.action_in(noisy_pred_action),
            self.action_in(gt_action_cond),
            self.horizon_embed(horizon_idx.long())[:, None, :],
            self.state_in(noisy_future_state),
            self.value_in(noisy_value),
            self.img_in(noisy_future_latents),
        ]
        segment_ids = torch.cat(
            [torch.full((batch_size, part.shape[1]), index, device=device, dtype=torch.long) for index, part in enumerate(parts)],
            dim=1,
        )
        img = torch.cat(parts, dim=1) + self.segment_embed(segment_ids)
        if noisy_future_dino is not None:
            dino_part = self.dino_in(noisy_future_dino) + self.dino_segment_embed.weight[0]
            img = torch.cat([img, dino_part], dim=1)

        id_dtype = current_ids.dtype
        action_time = torch.arange(1, noisy_pred_action.shape[1] + 1, device=device, dtype=id_dtype)[None]
        action_time = action_time.expand(batch_size, -1)
        gt_time = torch.arange(1, gt_action_cond.shape[1] + 1, device=device, dtype=id_dtype)[None]
        gt_time = gt_time.expand(batch_size, -1)
        nontext_ids = torch.cat(
            [
                self._robot_ids(batch_size=batch_size, length=state.shape[1], segment_id=1, device=device, dtype=id_dtype),
                current_ids.to(device=device),
                self._robot_ids(batch_size=batch_size, length=noisy_pred_action.shape[1], segment_id=3, device=device, dtype=id_dtype, time_ids=action_time),
                self._robot_ids(batch_size=batch_size, length=gt_action_cond.shape[1], segment_id=4, device=device, dtype=id_dtype, time_ids=gt_time),
                self._robot_ids(batch_size=batch_size, length=1, segment_id=5, device=device, dtype=id_dtype, time_ids=horizon_idx[:, None]),
                self._robot_ids(batch_size=batch_size, length=noisy_future_state.shape[1], segment_id=6, device=device, dtype=id_dtype),
                self._robot_ids(batch_size=batch_size, length=noisy_value.shape[1], segment_id=7, device=device, dtype=id_dtype),
                future_ids.to(device=device),
                *([] if dino_ids is None else [dino_ids.to(device=device)]),
            ],
            dim=1,
        )

        pe_img = self.pe_embedder(nontext_ids)
        pe_txt = self.pe_embedder(context_ids.to(device=device))
        bias = build_attention_bias(
            segments,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            horizon_idx=horizon_idx,
            context_mask=context_mask,
        )

        zeros = torch.zeros_like(wm_timestep)
        vec_clean = self._condition_vec(zeros, guidance)
        vec_action = self._condition_vec(action_timestep, guidance)
        vec_wm = self._condition_vec(wm_timestep, guidance)

        double_clean = self.double_stream_modulation_img(vec_clean)
        double_action = self.double_stream_modulation_img(vec_action)
        double_wm = self.double_stream_modulation_img(vec_wm)
        double_img = _stitch_double(
            [
                (lengths["state"], double_clean),
                (lengths["ref_image"], double_clean),
                (lengths["pred_action"], double_action),
                (lengths["gt_action"], double_clean),
                (lengths["horizon"], double_clean),
                (lengths["future_state"], double_wm),
                (lengths["value"], double_wm),
                (lengths["future_image"], double_wm),
                (lengths["future_dino"], double_wm),
            ]
        )
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
        single_mod = _stitch_triple(
            [
                (lengths["language"], single_clean),
                (lengths["state"], single_clean),
                (lengths["ref_image"], single_clean),
                (lengths["pred_action"], single_action),
                (lengths["gt_action"], single_clean),
                (lengths["horizon"], single_clean),
                (lengths["future_state"], single_wm),
                (lengths["value"], single_wm),
                (lengths["future_image"], single_wm),
                (lengths["future_dino"], single_wm),
            ]
        )
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
        action_hidden = hidden[:, segments.pred_action]
        state_hidden = hidden[:, segments.future_state]
        value_hidden = hidden[:, segments.value]
        dino_hidden = hidden[:, segments.future_dino]
        return Flux2FACTOutput(
            image=self.final_layer(image_hidden, vec_wm),
            action=self.action_out(action_hidden),
            future_state=self.state_out(state_hidden),
            value=self.value_out(value_hidden),
            dino=self.dino_out(dino_hidden) if self.dino_dim is not None and dino_hidden.shape[1] else None,
            segments=segments,
        )
