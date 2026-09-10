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
from .flux2_scalar_expert import DeterministicFlux2ScalarExpert, initialize_scalar_expert_from_flux


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
