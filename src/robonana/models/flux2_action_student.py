"""中文：独立一步学生；FLUX/Q/V state dict 与生产采样接口完全不变。
English: Standalone one-step action expert; never registered on the teacher.

Blocks, mixed attention and initialization reuse flux2_scalar_expert, adapted
from ImageWAM action_dit_flux2.py / mot.py / preprocess_action_dit_flux2.py:
https://github.com/yuyangalin/ImageWAM/tree/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a
Unlike ImageWAM flow prediction, this head predicts the final action directly.
Distillation target follows MAC agents/mac.py::bc_actor_loss (same noise):
https://github.com/kwanyoungpark/MAC/blob/main/agents/mac.py#L70-L101
"""
import torch
from torch import nn
from flux2.model import timestep_embedding
from .flux2_scalar_expert import (DeterministicFlux2ScalarExpert, initialize_scalar_expert_from_flux,
                                 _mixed_query_attention)


class OneStepActionExpert(DeterministicFlux2ScalarExpert):
    def __init__(self, *, action_dim=14, horizon=48, **kwargs):
        super().__init__(**kwargs)
        del self.query
        self.action_dim, self.horizon = int(action_dim), int(horizon)
        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim, bias=False)
        self.head.linear = nn.Linear(self.hidden_dim, self.action_dim, bias=False)

    def forward(self, cache, *, noise, query_pe):
        if tuple(noise.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError("student noise must be [batch, 48, action_dim]")
        # C-only cache: reject a clean-action branch (teacher-label leakage).
        if cache.parent is not None:
            raise ValueError("action student accepts only the L/S/I condition cache")
        if any(t.requires_grad for stream in ("double", "single")
               for layer in cache.layers(stream) for t in layer.values()):
            raise ValueError("teacher cache must be detached")
        query = self.action_encoder(noise.to(self.action_encoder.weight.dtype))
        zeros = torch.zeros(noise.shape[0], device=noise.device, dtype=torch.float32)
        vec = self.time_in(timestep_embedding(zeros, 256).to(self.action_encoder.weight.dtype))
        return self.forward_tokens(cache, query=query, query_pe=query_pe, vec=vec)


def build_action_student(teacher, hidden_dim=1024):
    """Fresh task encoder/head; FLUX blocks/modulation use existing transfer."""
    student = OneStepActionExpert(action_dim=teacher.action_dim, horizon=teacher.chunk_horizon,
        hidden_dim=hidden_dim, num_heads=teacher.num_heads,
        attn_head_dim=teacher.hidden_size // teacher.num_heads,
        num_layers_double=len(teacher.double_blocks), num_layers_single=len(teacher.single_blocks),
        mlp_ratio=teacher.double_blocks[0].img_mlp[0].out_features / (2 * teacher.hidden_size))
    initialize_scalar_expert_from_flux(student, teacher)
    return student


class FlowActionExpert(OneStepActionExpert):
    """V3 flow branch: shared slim blocks, real diffusion time, attached C K/V.

    This is not the one-step distillation interface: training must preserve
    the path from action loss through condition K/V into the FLUX backbone.
    The inherited encoder/head are fresh; body transfer uses the same helper
    as the scalar experts. No extra learned horizon/segment token is added.
    """

    def prepare(self, action, timestep):
        if action.ndim != 3 or action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("flow action must be [batch,48,action_dim]")
        if timestep.shape != (action.shape[0],):
            raise ValueError("flow timestep must be [batch]")
        dtype = self.action_encoder.weight.dtype
        query = self.action_encoder(action.to(dtype))
        vec = self.time_in(timestep_embedding(timestep, 256).to(dtype))
        return query, vec

    def advance(self, block, query, query_pe, modulation, condition_k, condition_v, key_mask):
        # One softmax over C + A. Never detach C: action BC also trains FLUX.
        state = block.prepare_qkv(query, query_pe, modulation)
        attention = _mixed_query_attention(
            state["q"], torch.cat((condition_k, state["k"]), dim=1),
            torch.cat((condition_v, state["v"]), dim=1),
            num_heads=self.num_heads, head_dim=self.attn_head_dim, key_mask=key_mask,
        )
        return block.apply_post(attention, state)

    def forward(self, cache, *, action, timestep, query_pe):
        if cache.parent is not None:
            raise ValueError("flow action requires a condition-only cache")
        query, vec = self.prepare(action, timestep)
        return self.forward_tokens(cache, query=query, query_pe=query_pe, vec=vec)
