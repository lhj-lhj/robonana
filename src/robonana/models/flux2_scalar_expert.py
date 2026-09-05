"""Deterministic one-query FLUX.2 experts for MAC Value and Q.

The slim block structure and cached cross-expert attention are adapted from
ImageWAM's FLUX.2 action expert at the pinned revision below:

* https://github.com/yuyangalin/ImageWAM/blob/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a/src/imagewam/models/backbones/action_dit_flux2.py
* https://github.com/yuyangalin/ImageWAM/blob/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a/src/imagewam/models/backbones/mot.py#L612-L745
* https://github.com/yuyangalin/ImageWAM/blob/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a/scripts/flux2/preprocess_action_dit_flux2.py

ImageWAM is MIT licensed.  RoboNana deliberately keeps only a thin adapter:
there is no copied trainer, dataset, diffusion scheduler, or action-flow head.
Each expert owns exactly one learned query and predicts one deterministic
scalar after attending to per-layer K/V produced by the frozen FLUX backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from flux2.model import MLPEmbedder, Modulation, QKNorm, SiLUActivation, apply_rope, timestep_embedding


@dataclass(frozen=True)
class FrozenFluxKVCache:
    """Detached per-layer K/V from one frozen FLUX prefix."""

    double: tuple[dict[str, Tensor], ...]
    single: tuple[dict[str, Tensor], ...]
    key_mask: Tensor
    prefix_length: int


def _flatten_heads(value: Tensor) -> Tensor:
    return value.transpose(1, 2).reshape(
        value.shape[0], value.shape[2], value.shape[1] * value.shape[3]
    )


def _mixed_query_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    num_heads: int,
    head_dim: int,
    key_mask: Tensor,
) -> Tensor:
    """Run ImageWAM-style mixed attention for expert queries only."""

    batch, query_length, _ = q.shape
    key_length = k.shape[1]
    q = q.view(batch, query_length, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, key_length, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, key_length, num_heads, head_dim).transpose(1, 2)
    mask = key_mask[:, None, None, :].expand(batch, 1, query_length, key_length)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return out.transpose(1, 2).reshape(batch, query_length, num_heads * head_dim)


class SlimFlux2SelfAttention(nn.Module):
    """ImageWAM-compatible slim attention with full FLUX head width."""

    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.attn_dim, bias=False)
        self.norm = QKNorm(self.attn_head_dim)
        self.proj = nn.Linear(self.attn_dim, self.hidden_dim, bias=False)


class SlimFlux2DoubleBlock(nn.Module):
    """The image branch of ImageWAM's slim FLUX.2 double block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_dim = int(num_heads) * int(attn_head_dim)
        mlp_hidden_dim = int(round(hidden_dim * float(mlp_ratio)))
        self.img_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.img_attn = SlimFlux2SelfAttention(hidden_dim, num_heads, attn_head_dim)
        self.img_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim * 2, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_dim, bias=False),
        )

    def prepare_qkv(self, x: Tensor, pe: Tensor, modulation) -> dict[str, Tensor]:
        mod1, mod2 = modulation
        shift1, scale1, gate1 = mod1
        shift2, scale2, gate2 = mod2
        x_mod = (1 + scale1) * self.img_norm1(x) + shift1
        qkv = self.img_attn.qkv(x_mod)
        q, k, v = rearrange(
            qkv, "b l (three h d) -> three b h l d", three=3, h=self.num_heads
        )
        q, k = self.img_attn.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return {
            "q": _flatten_heads(q),
            "k": _flatten_heads(k),
            "v": _flatten_heads(v),
            "residual": x,
            "shift2": shift2,
            "scale2": scale2,
            "gate1": gate1,
            "gate2": gate2,
        }

    def apply_post(self, attention: Tensor, state: dict[str, Tensor]) -> Tensor:
        x = state["residual"] + state["gate1"] * self.img_attn.proj(attention)
        return x + state["gate2"] * self.img_mlp(
            (1 + state["scale2"]) * self.img_norm2(x) + state["shift2"]
        )


class SlimFlux2SingleBlock(nn.Module):
    """ImageWAM's slim FLUX.2 single block, unchanged in computation."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_dim = int(num_heads) * int(attn_head_dim)
        self.mlp_hidden_dim = int(round(hidden_dim * float(mlp_ratio)))
        self.linear1 = nn.Linear(
            hidden_dim,
            3 * self.attn_dim + self.mlp_hidden_dim * 2,
            bias=False,
        )
        self.linear2 = nn.Linear(self.attn_dim + self.mlp_hidden_dim, hidden_dim, bias=False)
        self.norm = QKNorm(attn_head_dim)
        self.pre_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

    def prepare_qkv(self, x: Tensor, pe: Tensor, modulation) -> dict[str, Tensor]:
        shift, scale, gate = modulation
        x_mod = (1 + scale) * self.pre_norm(x) + shift
        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.attn_dim, self.mlp_hidden_dim * 2],
            dim=-1,
        )
        q, k, v = rearrange(
            qkv, "b l (three h d) -> three b h l d", three=3, h=self.num_heads
        )
        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return {
            "q": _flatten_heads(q),
            "k": _flatten_heads(k),
            "v": _flatten_heads(v),
            "mlp": mlp,
            "gate": gate,
            "residual": x,
        }

    def apply_post(self, attention: Tensor, state: dict[str, Tensor]) -> Tensor:
        output = self.linear2(torch.cat((attention, self.mlp_act(state["mlp"])), dim=2))
        return state["residual"] + state["gate"] * output


class Flux2ScalarHead(nn.Module):
    """ImageWAM ``Flux2ActionHead`` specialized to one scalar."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, 1, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear((1 + scale[:, None]) * self.norm_final(x) + shift[:, None])


class DeterministicFlux2ScalarExpert(nn.Module):
    """One learned query that reads a frozen FLUX prefix and returns a scalar."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        num_layers_double: int,
        num_layers_single: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.query = nn.Embedding(1, hidden_dim)
        self.time_in = MLPEmbedder(256, hidden_dim, disable_bias=True)
        self.double_stream_modulation_img = Modulation(hidden_dim, double=True, disable_bias=True)
        self.single_stream_modulation = Modulation(hidden_dim, double=False, disable_bias=True)
        self.double_blocks = nn.ModuleList(
            SlimFlux2DoubleBlock(
                hidden_dim, num_heads, attn_head_dim, mlp_ratio
            )
            for _ in range(num_layers_double)
        )
        self.single_blocks = nn.ModuleList(
            SlimFlux2SingleBlock(
                hidden_dim, num_heads, attn_head_dim, mlp_ratio
            )
            for _ in range(num_layers_single)
        )
        self.head = Flux2ScalarHead(hidden_dim)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
        # ``nn.Embedding.reset_parameters`` uses unit variance, which is far
        # too large beside interpolated FLUX activations. Keep the only new
        # token on the same small-query scale used by transformer adapters.
        nn.init.normal_(self.query.weight, std=0.02)

    def forward(
        self,
        cache: FrozenFluxKVCache,
        *,
        query_pe: Tensor,
    ) -> Tensor:
        batch = cache.key_mask.shape[0]
        if query_pe.shape[0] != batch or query_pe.shape[2] != 1:
            raise ValueError("expert query_pe must contain exactly one query per batch item")
        dtype = self.query.weight.dtype
        query = self.query.weight[None].expand(batch, 1, -1).to(dtype=dtype)
        zeros = torch.zeros(batch, device=query.device, dtype=torch.float32)
        vec = self.time_in(timestep_embedding(zeros, 256).to(dtype=dtype))
        double_mod = self.double_stream_modulation_img(vec)
        single_mod = self.single_stream_modulation(vec)[0]
        expert_key_mask = torch.cat(
            [cache.key_mask.to(device=query.device), torch.ones(batch, 1, device=query.device, dtype=torch.bool)],
            dim=1,
        )

        for block, layer_cache in zip(self.double_blocks, cache.double, strict=True):
            state = block.prepare_qkv(query, query_pe, double_mod)
            k = torch.cat([layer_cache["k"].to(dtype=state["k"].dtype), state["k"]], dim=1)
            v = torch.cat([layer_cache["v"].to(dtype=state["v"].dtype), state["v"]], dim=1)
            mixed = _mixed_query_attention(
                state["q"], k, v,
                num_heads=self.num_heads,
                head_dim=self.attn_head_dim,
                key_mask=expert_key_mask,
            )
            query = block.apply_post(mixed, state)

        for block, layer_cache in zip(self.single_blocks, cache.single, strict=True):
            state = block.prepare_qkv(query, query_pe, single_mod)
            k = torch.cat([layer_cache["k"].to(dtype=state["k"].dtype), state["k"]], dim=1)
            v = torch.cat([layer_cache["v"].to(dtype=state["v"].dtype), state["v"]], dim=1)
            mixed = _mixed_query_attention(
                state["q"], k, v,
                num_heads=self.num_heads,
                head_dim=self.attn_head_dim,
                key_mask=expert_key_mask,
            )
            query = block.apply_post(mixed, state)
        return self.head(query, vec).squeeze(1)


def _resize_tensor_to_shape(source: Tensor, target_shape: tuple[int, ...]) -> Tensor:
    """ImageWAM's axis-wise linear interpolation used for slim expert init."""

    if tuple(source.shape) == target_shape:
        return source
    output = source.float()
    while output.ndim < len(target_shape):
        output = output.unsqueeze(0)
    while output.ndim > len(target_shape):
        if output.shape[0] != 1:
            raise ValueError(
                f"cannot reduce source shape {tuple(source.shape)} to {target_shape}"
            )
        output = output.squeeze(0)
    for dimension, new_size in enumerate(target_shape):
        if output.shape[dimension] == new_size:
            continue
        permutation = [index for index in range(output.ndim) if index != dimension] + [dimension]
        inverse = [0] * output.ndim
        for index, original in enumerate(permutation):
            inverse[original] = index
        transposed = output.permute(*permutation).contiguous()
        flat = transposed.reshape(-1, 1, transposed.shape[-1])
        flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
        output = flat.reshape(*transposed.shape[:-1], new_size).permute(*inverse).contiguous()
    return output.to(dtype=source.dtype)


@torch.no_grad()
def initialize_scalar_expert_from_flux(
    expert: DeterministicFlux2ScalarExpert,
    flux: nn.Module,
    *,
    alpha_scaling: bool = True,
) -> tuple[int, int]:
    """Initialize the slim expert from the matching frozen FLUX branches.

    This follows ImageWAM's preprocessing policy: exact copies where shapes
    match, axis-wise interpolation otherwise, and fan-in alpha correction when
    the last dimension changes.  The learned query is intentionally new.
    """

    source_state = flux.state_dict()
    target_state = expert.state_dict()
    copied = 0
    resized = 0
    mapped: dict[str, Tensor] = {}
    for name, target in target_state.items():
        if name.startswith("query."):
            continue
        if name.startswith("head."):
            source_name = "final_layer." + name.removeprefix("head.")
        else:
            source_name = name
        source = source_state.get(source_name)
        if source is None:
            continue
        if tuple(source.shape) == tuple(target.shape):
            value = source
            copied += 1
        else:
            value = _resize_tensor_to_shape(source, tuple(target.shape))
            if alpha_scaling and source.ndim >= 2 and source.shape[-1] != target.shape[-1]:
                value = value.float() * (float(source.shape[-1]) / float(target.shape[-1])) ** 0.5
            resized += 1
        mapped[name] = value.to(device=target.device, dtype=target.dtype)
    incompatible = expert.load_state_dict(mapped, strict=False)
    expected_missing = {"query.weight"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "FLUX-to-expert initialization mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return copied, resized
