"""A thin shared-backbone extension of the official FLUX.2 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from flux2.model import Flux2, Flux2Params, apply_rope, timestep_embedding

from .attention_mask import MacSegmentMap


@dataclass
class Flux2FACTOutput:
    image: Tensor
    action: Tensor
    future_state: Tensor
    reward: Tensor
    success: Tensor
    q: Tensor | None
    dino: Tensor | None
    segments: MacSegmentMap
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
        # Keep the eight-row embedding used by the official FACT backbone.
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

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use MacFlux2FACTModel; this base only supplies shared FLUX modules/helpers.")
