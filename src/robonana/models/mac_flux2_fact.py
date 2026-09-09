"""Current fixed-48 MAC actor/world model with separate MoT scalar experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from flux2.model import Flux2Params, apply_rope

from .attention_mask import (
    MacSegmentMap,
    build_mac_attention_bias,
    build_mac_critic_prefix_bias,
)
from .flux2_fact import (
    Flux2FACTModel,
    Flux2FACTOutput,
    _masked_attention,
    _stitch_double,
    _stitch_triple,
)
from .flux2_scalar_expert import (
    DeterministicFlux2ScalarExpert,
    FrozenFluxKVCache,
    _flatten_heads,
    _mixed_query_attention,
)


@dataclass(frozen=True)
class FrozenWorldCache:
    """Request-local [C,G,R,U] K/V; never persist across FLUX updates."""

    kv: FrozenFluxKVCache
    segments: MacSegmentMap
    future_bias: Tensor
    reward: Tensor
    success: Tensor


class MacFlux2FACTModel(Flux2FACTModel):
    """Single FLUX actor/world backbone plus deterministic Value/Q experts.

    The shared sequence is exactly ``[L,S,I,A,G,R,U,S',I']``.  Neither critic
    query is inserted into it.  During critic training, the frozen FLUX prefix
    is evaluated under ``torch.no_grad`` and each scalar expert consumes its
    per-layer K/V using the ImageWAM-style cached MoT adapter implemented in
    :mod:`robonana.models.flux2_scalar_expert`.
    """

    architecture_version = "mac_mot_v2"

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
        expert_hidden_dim: int | None = None,
    ) -> None:
        if int(chunk_horizon) != 48:
            raise ValueError("mac_mot_v2 requires chunk_horizon=48")
        if int(reward_dim) != int(chunk_horizon):
            raise ValueError("mac_mot_v2 requires one reward logit per chunk step")
        if (int(success_dim), int(q_dim), int(value_dim)) != (1, 1, 1):
            raise ValueError("mac_mot_v2 success, Q, and Value outputs must be scalar")
        if dino_dim is not None:
            raise ValueError("mac_mot_v2 world sequence does not include a DINO target")
        super().__init__(
            params,
            action_dim=action_dim,
            state_dim=state_dim,
            reward_dim=1,
            success_dim=1,
            q_dim=1,
            max_horizon=chunk_horizon,
            dino_dim=None,
            pred_action_bidirectional=True,
        )
        self.chunk_horizon = int(chunk_horizon)
        self.max_horizon = self.chunk_horizon
        self.reward_dim = int(reward_dim)
        self.value_dim = int(value_dim)
        self.expert_hidden_dim = int(
            min(1024, self.hidden_size) if expert_hidden_dim is None else expert_hidden_dim
        )
        if self.expert_hidden_dim <= 0:
            raise ValueError("expert_hidden_dim must be positive")

        # Remove variable-horizon/flow-Q modules; MAC scalar heads below are new.
        del self.q_in
        del self.horizon_embed
        del self.segment_embed
        del self.q_segment_embed
        del self.q_out

        self.actor_world_segment_embed = nn.Embedding(8, self.hidden_size)
        self.reward_out = nn.Linear(self.hidden_size, reward_dim, bias=False)
        head_dim = self.hidden_size // self.num_heads
        expert_kwargs = dict(
            hidden_dim=self.expert_hidden_dim,
            num_heads=self.num_heads,
            attn_head_dim=head_dim,
            num_layers_double=len(self.double_blocks),
            num_layers_single=len(self.single_blocks),
            mlp_ratio=float(params.mlp_ratio),
        )
        self.value_expert = DeterministicFlux2ScalarExpert(**expert_kwargs)
        self.q_expert = DeterministicFlux2ScalarExpert(**expert_kwargs)
        self.value_expert.reset_parameters()
        self.q_expert.reset_parameters()
        self._mac_training_phase = "world_policy"
        # 中文：仅控制训练激活重计算，不改变权重、attention 或推理。
        # English: Stride 1 checkpoints every single block; stride 2 keeps
        # even-index blocks only. Double blocks retain the existing policy.
        self.gradient_checkpointing_single_stride = 1

    def set_gradient_checkpointing_single_stride(self, stride: int) -> None:
        if type(stride) is not int or stride < 1:
            raise ValueError("gradient_checkpointing_single_stride must be a positive integer")
        self.gradient_checkpointing_single_stride = stride

    def set_training_phase(self, phase: str) -> tuple[str, ...]:
        """Select the only two supported optimizer surfaces."""

        if phase not in {"world_policy", "critic"}:
            raise ValueError("MAC phase must be world_policy or critic")
        self._mac_training_phase = phase
        train_experts = phase == "critic"
        trainable: list[str] = []
        for name, parameter in self.named_parameters():
            is_expert = name.startswith(("value_expert.", "q_expert."))
            parameter.requires_grad_(is_expert == train_experts)
            if parameter.requires_grad:
                trainable.append(name)
        self.train(self.training)
        return tuple(trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_mac_training_phase", "world_policy") == "critic":
            # Model.training remains true for the surrounding trainer, while
            # the entire FLUX actor/world side is deterministic eval-only.
            for name, child in self.named_children():
                if name not in {"value_expert", "q_expert"}:
                    child.train(False)
            self.value_expert.train(mode)
            self.q_expert.train(mode)
        else:
            self.value_expert.train(False)
            self.q_expert.train(False)
        return self

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
        chunk_horizon: Tensor,
        noisy_future_state: Tensor,
        noisy_reward: Tensor,
        noisy_q: Tensor,
        action_timestep: Tensor,
        wm_timestep: Tensor,
        noisy_future_dino: Tensor | None = None,
        dino_ids: Tensor | None = None,
        context_mask: Tensor | None = None,
        guidance: Tensor | None = None,
        critic_kind: str | None = None,
        condition_cache: FrozenFluxKVCache | None = None,
    ) -> Flux2FACTOutput | Tensor:
        self.cache_compute_dtype()  # Reject unsupported weights/autocast early.
        if critic_kind is not None:
            if critic_kind == "both":
                # Only request-local FP32 caches may enter critic regression.
                cache = condition_cache if self.condition_cache_compatible(condition_cache) else None
                cache = cache if cache is not None else self.prefill_condition_cache(
                    context=context, context_ids=context_ids,
                    current_latents=current_latents, current_ids=current_ids,
                    state=state, context_mask=context_mask,
                )
                return (
                    self.predict_value(
                        context=context,
                        context_ids=context_ids,
                        current_latents=current_latents,
                        current_ids=current_ids,
                        state=state,
                        context_mask=context_mask,
                        cache=cache,
                    ),
                    self.predict_q(
                        context=context,
                        context_ids=context_ids,
                        current_latents=current_latents,
                        current_ids=current_ids,
                        state=state,
                        clean_action=gt_action_cond,
                        context_mask=context_mask,
                        cache=cache,
                    ),
                )
            if critic_kind == "value":
                return self.predict_value(
                    context=context,
                    context_ids=context_ids,
                    current_latents=current_latents,
                    current_ids=current_ids,
                    state=state,
                    context_mask=context_mask,
                )
            if critic_kind == "q":
                return self.predict_q(
                    context=context,
                    context_ids=context_ids,
                    current_latents=current_latents,
                    current_ids=current_ids,
                    state=state,
                    clean_action=gt_action_cond,
                    context_mask=context_mask,
                )
            raise ValueError("critic_kind must be both, value, q, or None")
        batch = context.shape[0]
        chunk_horizon = chunk_horizon.reshape(-1)
        if tuple(chunk_horizon.shape) != (batch,) or not bool(
            torch.all(chunk_horizon == self.chunk_horizon)
        ):
            raise ValueError("mac_mot_v2 actor/world forward requires chunk_horizon=48")
        del chunk_horizon, noisy_reward, noisy_q
        if noisy_future_dino is not None or dino_ids is not None:
            raise ValueError("mac_mot_v2 does not accept DINO future tokens")
        if context_ids.shape != (*context.shape[:2], 4):
            raise ValueError("context_ids must have shape [B, text_tokens, 4]")
        if current_ids.shape != (*current_latents.shape[:2], 4):
            raise ValueError("current_ids must have shape [B, image_tokens, 4]")
        if future_ids.shape != (*noisy_future_latents.shape[:2], 4):
            raise ValueError("future_ids must have shape [B, image_tokens, 4]")
        for name, value, width in (
            ("state", state, self.state_dim),
            ("noisy_future_state", noisy_future_state, self.state_dim),
            ("noisy_pred_action", noisy_pred_action, self.action_dim),
            ("gt_action_cond", gt_action_cond, self.action_dim),
        ):
            if value.ndim != 3 or value.shape[0] != batch or value.shape[-1] != width:
                raise ValueError(f"{name} must have shape [B, tokens, {width}]")
        if noisy_pred_action.shape[1] not in (0, self.chunk_horizon):
            raise ValueError("predicted action must be empty or one 48-step chunk")
        if gt_action_cond.shape[1] not in (0, self.chunk_horizon):
            raise ValueError("clean action must be empty or one 48-step chunk")

        dtype = self.img_in.weight.dtype
        device = context.device
        context = context.to(dtype=dtype)
        current_latents = current_latents.to(dtype=dtype)
        noisy_future_latents = noisy_future_latents.to(dtype=dtype)
        state = state.to(dtype=dtype)
        noisy_pred_action = noisy_pred_action.to(dtype=dtype)
        gt_action_cond = gt_action_cond.to(dtype=dtype)
        noisy_future_state = noisy_future_state.to(dtype=dtype)
        segments = MacSegmentMap.from_lengths(
            language=context.shape[1],
            state=state.shape[1],
            ref_image=current_latents.shape[1],
            pred_action=noisy_pred_action.shape[1],
            clean_action=gt_action_cond.shape[1],
            reward=1,
            success=1,
            future_state=noisy_future_state.shape[1],
            future_image=noisy_future_latents.shape[1],
        )

        def tagged(value: Tensor, segment: int) -> Tensor:
            return value + self.actor_world_segment_embed.weight[segment]

        txt = self.txt_in(context)
        img = torch.cat(
            [
                tagged(self.state_in(state), 0),
                tagged(self.img_in(current_latents), 1),
                tagged(self.action_in(noisy_pred_action), 2),
                tagged(self.action_in(gt_action_cond), 3),
                tagged(self.reward_token.weight[None].expand(batch, 1, -1), 4),
                tagged(self.success_token.weight[None].expand(batch, 1, -1), 5),
                tagged(self.state_in(noisy_future_state), 6),
                tagged(self.img_in(noisy_future_latents), 7),
            ],
            dim=1,
        )
        id_dtype = current_ids.dtype
        action_time = torch.arange(1, noisy_pred_action.shape[1] + 1, device=device, dtype=id_dtype)[None].expand(batch, -1)
        clean_time = torch.arange(1, gt_action_cond.shape[1] + 1, device=device, dtype=id_dtype)[None].expand(batch, -1)
        ids = torch.cat(
            [
                self._robot_ids(batch_size=batch, length=state.shape[1], segment_id=1, device=device, dtype=id_dtype),
                current_ids.to(device=device),
                self._robot_ids(batch_size=batch, length=noisy_pred_action.shape[1], segment_id=3, device=device, dtype=id_dtype, time_ids=action_time),
                self._robot_ids(batch_size=batch, length=gt_action_cond.shape[1], segment_id=4, device=device, dtype=id_dtype, time_ids=clean_time),
                self._robot_ids(batch_size=batch, length=1, segment_id=5, device=device, dtype=id_dtype),
                self._robot_ids(batch_size=batch, length=1, segment_id=6, device=device, dtype=id_dtype),
                self._robot_ids(batch_size=batch, length=noisy_future_state.shape[1], segment_id=7, device=device, dtype=id_dtype),
                future_ids.to(device=device),
            ],
            dim=1,
        )
        pe_img = self.pe_embedder(ids)
        pe_txt = self.pe_embedder(context_ids.to(device=device))
        bias = build_mac_attention_bias(
            segments,
            batch_size=batch,
            dtype=dtype,
            device=device,
            context_mask=context_mask,
        )

        zero = torch.zeros_like(wm_timestep)
        vec_clean = self._condition_vec(zero, guidance)
        vec_action = self._condition_vec(action_timestep, guidance)
        vec_world = self._condition_vec(wm_timestep, guidance)
        clean_double = self.double_stream_modulation_img(vec_clean)
        action_double = self.double_stream_modulation_img(vec_action)
        world_double = self.double_stream_modulation_img(vec_world)
        lengths = (
            state.shape[1], current_latents.shape[1], noisy_pred_action.shape[1],
            gt_action_cond.shape[1], 1, 1, noisy_future_state.shape[1],
            noisy_future_latents.shape[1],
        )
        double_img = _stitch_double(zip(
            lengths,
            (clean_double, clean_double, action_double, clean_double, clean_double, clean_double, world_double, world_double),
            strict=True,
        ))
        double_txt = self.double_stream_modulation_txt(vec_clean)
        for block in self.double_blocks:
            if self.gradient_checkpointing and self.training:
                img, txt = checkpoint(
                    lambda img_, txt_, block_=block: self._double_block_forward(
                        block_, img_, txt_, pe_img, pe_txt, double_img, double_txt, bias
                    ),
                    img, txt, use_reentrant=False,
                )
            else:
                img, txt = self._double_block_forward(
                    block, img, txt, pe_img, pe_txt, double_img, double_txt, bias
                )

        hidden = torch.cat([txt, img], dim=1)
        pe = torch.cat([pe_txt, pe_img], dim=2)
        clean_single = self.single_stream_modulation(vec_clean)[0]
        action_single = self.single_stream_modulation(vec_action)[0]
        world_single = self.single_stream_modulation(vec_world)[0]
        single_mod = _stitch_triple(zip(
            (context.shape[1], *lengths),
            (clean_single, clean_single, clean_single, action_single, clean_single, clean_single, clean_single, world_single, world_single),
            strict=True,
        ))
        for block_index, block in enumerate(self.single_blocks):
            if (self.gradient_checkpointing and self.training
                    and block_index % self.gradient_checkpointing_single_stride == 0):
                hidden = checkpoint(
                    lambda hidden_, block_=block: self._single_block_forward(
                        block_, hidden_, pe, single_mod, bias
                    ),
                    hidden, use_reentrant=False,
                )
            else:
                hidden = self._single_block_forward(block, hidden, pe, single_mod, bias)

        return Flux2FACTOutput(
            image=self.final_layer(hidden[:, segments.future_image], vec_world),
            action=self.action_out(hidden[:, segments.pred_action]),
            future_state=self.state_out(hidden[:, segments.future_state]),
            reward=self.reward_out(hidden[:, segments.reward]).squeeze(1),
            success=self.success_out(hidden[:, segments.success]).squeeze(1),
            q=None,
            dino=None,
            segments=segments,
            value=None,
        )

    def prefill_condition_cache(self, **kwargs) -> FrozenFluxKVCache:
        """Compute C once per observation/weight snapshot at clean timestep 0.

        Callers own this ephemeral cache. Never attach it to the model or
        reuse it after an observation, guidance, dtype, or FLUX weight change.
        """
        return self.prefill_critic_cache(**kwargs, clean_action=None)

    def cache_compute_dtype(self):
        device_type = self.img_in.weight.device.type
        if self.img_in.weight.dtype != torch.float32 or torch.is_autocast_enabled(device_type):
            raise ValueError("MAC requires FP32 weights with autocast disabled")
        return torch.float32

    def condition_cache_compatible(self, cache):
        return (cache is not None and cache.parent is None
                and cache.compute_dtype == self.cache_compute_dtype()
                and cache.key_mask.device == self.img_in.weight.device)

    @torch.no_grad()
    def _world_suffix(self, cache, hidden, ids, timestep, bias, *, capture):
        """Thin cached adapter over official FLUX blocks, not a new transformer.

        Same prepare/RoPE/mixed-attention/residual order as ImageWAM MoT:
        https://github.com/yuyangalin/ImageWAM/blob/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a/src/imagewam/models/backbones/mot.py#L612-L745
        Unlike Q's single query, world queries need a rectangular graph mask:
        R cannot read U; S' cannot read I'. All visible keys share ONE softmax.
        """
        if cache.compute_dtype != self.cache_compute_dtype():
            raise ValueError("world cache precision differs from current execution precision")
        batch = hidden.shape[0]
        pe = self.pe_embedder(ids)
        txt, pe_txt = hidden[:, :0], pe[:, :, :0]
        vec = self._condition_vec(timestep.expand(batch), None)
        mod_img = self.double_stream_modulation_img(vec)
        mod_txt = self.double_stream_modulation_txt(vec)

        def attend(q, k, v, shared):
            def heads(value):
                return value.reshape(batch, -1, self.num_heads,
                                     self.hidden_size // self.num_heads).transpose(1, 2)
            return _masked_attention(q, torch.cat((heads(shared["k"]), k), dim=2),
                                     torch.cat((heads(shared["v"]), v), dim=2), bias)

        double, single = [], []
        for block, shared in zip(self.double_blocks, cache.layers("double"), strict=True):
            q, k, v, full_pe, _, mods = block._prepare_qkv(
                hidden, txt, pe, pe_txt, mod_img, mod_txt)
            q, k = apply_rope(q, k, full_pe)
            if capture:
                double.append({"k": _flatten_heads(k).detach(), "v": _flatten_heads(v).detach()})
            attn = attend(q, k, v, shared)
            hidden, txt = block._apply_residuals(hidden, txt, attn, attn[:, :0], mods)
        mod = self.single_stream_modulation(vec)[0]
        for block, shared in zip(self.single_blocks, cache.layers("single"), strict=True):
            q, k, v, mlp, gate = block._qkv(hidden, mod)
            q, k = apply_rope(q, k, pe)
            if capture:
                single.append({"k": _flatten_heads(k).detach(), "v": _flatten_heads(v).detach()})
            hidden = block._out(hidden, attend(q, k, v, shared), mlp, gate)
        branch = None
        if capture:
            key_mask = torch.cat((cache.key_mask, torch.ones(
                batch, ids.shape[1], device=ids.device, dtype=torch.bool)), dim=1)
            branch = FrozenFluxKVCache(
                double=tuple(double), single=tuple(single), key_mask=key_mask,
                prefix_length=key_mask.shape[1], parent=cache,
                batch_indices=torch.arange(batch, device=ids.device),
                compute_dtype=cache.compute_dtype)
        return hidden, branch

    @torch.no_grad()
    def prefill_world_cache(self, *, condition_cache, clean_action, language_length,
                            state_length, image_length, future_state_length,
                            future_image_length, context_mask):
        """Compute clean G/R/U once; reuse C from this action-selection request.

        Reward/success are deterministic zero-time tokens, not teacher-forced
        labels. Their logits and layer K/V cannot depend on future noise.
        """
        if not self.condition_cache_compatible(condition_cache):
            raise ValueError("world prefill requires a same-precision condition-only cache")
        batch, device = clean_action.shape[0], clean_action.device
        if clean_action.shape != (batch, self.chunk_horizon, self.action_dim):
            raise ValueError("world prefill requires one clean 48-step action chunk")
        segments = MacSegmentMap.from_lengths(
            language=language_length, state=state_length, ref_image=image_length,
            pred_action=0, clean_action=self.chunk_horizon, reward=1, success=1,
            future_state=future_state_length, future_image=future_image_length)
        if condition_cache.prefix_length != segments.clean_action.start:
            raise ValueError("world cache condition length mismatch")
        bias = build_mac_attention_bias(segments, batch_size=batch,
            dtype=self.img_in.weight.dtype, device=device, context_mask=context_mask)
        embed = self.actor_world_segment_embed.weight
        hidden = torch.cat((self.action_in(clean_action.to(self.img_in.weight.dtype)) + embed[3],
            self.reward_token.weight[None].expand(batch, 1, -1) + embed[4],
            self.success_token.weight[None].expand(batch, 1, -1) + embed[5]), dim=1)
        def robot(length, segment, time_ids=None):
            return self._robot_ids(batch_size=batch, length=length, segment_id=segment,
                device=device, dtype=torch.long, time_ids=time_ids)
        ids = torch.cat((robot(48, 4, torch.arange(1, 49, device=device)[None].expand(batch, -1)),
                         robot(1, 5), robot(1, 6)), dim=1)
        stop = segments.future_state.start
        hidden, kv = self._world_suffix(condition_cache, hidden, ids,
            torch.zeros(batch, device=device), bias[:, :, segments.clean_action.start:stop, :stop],
            capture=True)
        return FrozenWorldCache(kv, segments, bias[:, :, stop:, :].contiguous(),
                                self.reward_out(hidden[:, -2]), self.success_out(hidden[:, -1]))

    @torch.no_grad()
    def predict_world_cached(self, cache, *, noisy_future_latents, noisy_future_state,
                             future_ids, wm_timestep):
        """Advance only [S',I']; the 20-step Euler schedule remains unchanged."""
        batch, length = noisy_future_state.shape[:2]
        segments = cache.segments
        if (length != segments.future_state.stop - segments.future_state.start
                or noisy_future_latents.shape[1] != segments.future_image.stop - segments.future_image.start):
            raise ValueError("world cache future shape mismatch")
        dtype = self.img_in.weight.dtype
        hidden = torch.cat((self.state_in(noisy_future_state.to(dtype)) + self.actor_world_segment_embed.weight[6],
                            self.img_in(noisy_future_latents.to(dtype)) + self.actor_world_segment_embed.weight[7]), dim=1)
        ids = torch.cat((self._robot_ids(batch_size=batch, length=length, segment_id=7,
            device=hidden.device, dtype=future_ids.dtype), future_ids), dim=1)
        hidden, _ = self._world_suffix(cache.kv, hidden, ids, wm_timestep, cache.future_bias, capture=False)
        vec = self._condition_vec(wm_timestep.expand(batch), None)
        return Flux2FACTOutput(
            image=self.final_layer(hidden[:, length:], vec),
            action=hidden.new_empty(batch, 0, self.action_dim),
            future_state=self.state_out(hidden[:, :length]), reward=cache.reward,
            success=cache.success, q=None, dino=None, segments=segments)

    @torch.no_grad()
    def _action_from_cache(
        self, cache: FrozenFluxKVCache, action: Tensor, *,
        batch_indices: Tensor, timestep: Tensor, clean: bool,
    ) -> tuple[Tensor, FrozenFluxKVCache | None]:
        """Advance only the action stream using official FLUX block methods.

        This follows ImageWAM's per-layer prepare_qkv -> mixed attention ->
        apply_post ordering, but uses the main FLUX action/image branch:
        https://github.com/yuyangalin/ImageWAM/blob/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a/src/imagewam/models/backbones/mot.py#L612-L745
        Each attention uses ONE softmax over [C,G]. K is cached after RoPE,
        V before residual update, matching both the full pass and Q expert.
        Predicted A and clean G have different segment IDs; Q must re-encode G.
        """
        self.cache_compute_dtype()
        batch = action.shape[0]
        if action.shape != (batch, self.chunk_horizon, self.action_dim):
            raise ValueError("cached action must have shape [batch,48,action_dim]")
        if cache.parent is not None or batch_indices.shape != (batch,):
            raise ValueError("action branch requires a condition-only cache and batch mapping")
        device = action.device
        action = action.to(dtype=self.img_in.weight.dtype)
        hidden = self.action_in(action) + self.actor_world_segment_embed.weight[3 if clean else 2]
        time_ids = torch.arange(1, self.chunk_horizon + 1, device=device)[None].expand(batch, -1)
        ids = self._robot_ids(
            batch_size=batch, length=self.chunk_horizon, segment_id=4 if clean else 3,
            device=device, dtype=torch.long, time_ids=time_ids,
        )
        pe = self.pe_embedder(ids)
        # Empty text lets us reuse the official two-stream routines without
        # recomputing any text tokens; C is already represented by cached K/V.
        txt = hidden[:, :0]
        pe_txt = pe[:, :, :0]
        vec = self._condition_vec(timestep.to(device=device).expand(batch), None)
        double_mod = self.double_stream_modulation_img(vec)
        txt_mod = self.double_stream_modulation_txt(vec)
        key_mask = torch.cat((cache.key_mask.index_select(0, batch_indices),
                              torch.ones(batch, self.chunk_horizon, device=device, dtype=torch.bool)), dim=1)

        def attend(q, k, v, shared):
            return _mixed_query_attention(
                _flatten_heads(q),
                torch.cat((shared["k"].index_select(0, batch_indices), _flatten_heads(k)), dim=1),
                torch.cat((shared["v"].index_select(0, batch_indices), _flatten_heads(v)), dim=1),
                num_heads=self.num_heads, head_dim=self.hidden_size // self.num_heads,
                key_mask=key_mask,
            )

        double_cache = []
        for block, shared in zip(self.double_blocks, cache.double, strict=True):
            q, k, v, full_pe, _, mods = block._prepare_qkv(
                hidden, txt, pe, pe_txt, double_mod, txt_mod
            )
            q, k = apply_rope(q, k, full_pe)
            if clean:
                double_cache.append({"k": _flatten_heads(k).detach(), "v": _flatten_heads(v).detach()})
            attention = attend(q, k, v, shared)
            hidden, txt = block._apply_residuals(hidden, txt, attention, attention[:, :0], mods)
        single_cache = []
        single_mod = self.single_stream_modulation(vec)[0]
        for block, shared in zip(self.single_blocks, cache.single, strict=True):
            q, k, v, mlp, gate = block._qkv(hidden, single_mod)
            q, k = apply_rope(q, k, pe)
            if clean:
                single_cache.append({"k": _flatten_heads(k).detach(), "v": _flatten_heads(v).detach()})
            hidden = block._out(hidden, attend(q, k, v, shared), mlp, gate)
        branch = None
        if clean:
            branch = FrozenFluxKVCache(
                double=tuple(double_cache), single=tuple(single_cache),
                key_mask=key_mask.detach(), prefix_length=key_mask.shape[1],
                parent=cache, batch_indices=batch_indices,
                compute_dtype=cache.compute_dtype,
            )
        return hidden, branch

    @torch.no_grad()
    def predict_action_cached(self, cache, action, *, batch_indices, timestep):
        hidden, _ = self._action_from_cache(
            cache, action, batch_indices=batch_indices, timestep=timestep, clean=False
        )
        return self.action_out(hidden)

    def predict_q_cached(self, cache, clean_action, *, batch_indices):
        _, branch = self._action_from_cache(
            cache, clean_action, batch_indices=batch_indices,
            timestep=torch.zeros((), device=clean_action.device), clean=True,
        )
        query_pe = self._expert_query_pe(
            batch=clean_action.shape[0], device=clean_action.device,
            dtype=torch.long, segment_id=11,
        )
        # Only frozen FLUX computation above is no-grad. Keep the expert in
        # autograd for critic training and in the surrounding DDP forward.
        return self.q_expert(branch, query_pe=query_pe)

    def score_q_candidates(self, cache, clean_actions, *, candidate_batch_size=8):
        if clean_actions.ndim != 4 or candidate_batch_size <= 0 or clean_actions.shape[1] == 0:
            raise ValueError("expected nonempty [B,M,48,A] actions and positive candidate_batch_size")
        batch, count = clean_actions.shape[:2]
        if cache.key_mask.shape[0] != batch:
            raise ValueError("condition cache batch must match candidate observations")
        scores = []
        for start in range(0, count, candidate_batch_size):
            group = clean_actions[:, start:start + candidate_batch_size]
            width = group.shape[1]
            indices = torch.arange(batch, device=group.device).repeat_interleave(width)
            scores.append(self.predict_q_cached(
                cache, group.reshape(batch * width, self.chunk_horizon, self.action_dim),
                batch_indices=indices,
            ).reshape(batch, width))
        return torch.cat(scores, dim=1)

    @torch.no_grad()
    def prefill_critic_cache(
        self,
        *,
        context: Tensor,
        context_ids: Tensor,
        current_latents: Tensor,
        current_ids: Tensor,
        state: Tensor,
        clean_action: Tensor | None,
        context_mask: Tensor | None = None,
        guidance: Tensor | None = None,
    ) -> FrozenFluxKVCache:
        """Cache ``C`` for Value or ``[C,G]`` for Q from frozen FLUX."""

        self.cache_compute_dtype()
        batch = context.shape[0]
        dtype = self.img_in.weight.dtype
        device = context.device
        context = context.to(dtype=dtype)
        current_latents = current_latents.to(dtype=dtype)
        state = state.to(dtype=dtype)
        action_length = 0 if clean_action is None else clean_action.shape[1]
        if clean_action is not None:
            if clean_action.shape != (batch, self.chunk_horizon, self.action_dim):
                raise ValueError(
                    f"Q clean_action must have shape {(batch, self.chunk_horizon, self.action_dim)}"
                )
            clean_action = clean_action.to(dtype=dtype)
        parts = [
            self.state_in(state) + self.actor_world_segment_embed.weight[0],
            self.img_in(current_latents) + self.actor_world_segment_embed.weight[1],
        ]
        id_parts = [
            self._robot_ids(batch_size=batch, length=state.shape[1], segment_id=1, device=device, dtype=current_ids.dtype),
            current_ids.to(device=device),
        ]
        if clean_action is not None:
            parts.append(self.action_in(clean_action) + self.actor_world_segment_embed.weight[3])
            action_time = torch.arange(1, action_length + 1, device=device, dtype=current_ids.dtype)[None].expand(batch, -1)
            id_parts.append(self._robot_ids(batch_size=batch, length=action_length, segment_id=4, device=device, dtype=current_ids.dtype, time_ids=action_time))
        img = torch.cat(parts, dim=1)
        txt = self.txt_in(context)
        img_ids = torch.cat(id_parts, dim=1)
        pe_img = self.pe_embedder(img_ids)
        pe_txt = self.pe_embedder(context_ids.to(device=device))
        bias, key_mask = build_mac_critic_prefix_bias(
            language_length=context.shape[1],
            state_length=state.shape[1],
            image_length=current_latents.shape[1],
            action_length=action_length,
            batch_size=batch,
            dtype=dtype,
            device=device,
            context_mask=context_mask,
        )
        zero = torch.zeros(batch, device=device, dtype=torch.float32)
        vec = self._condition_vec(zero, guidance)
        double_img = self.double_stream_modulation_img(vec)
        double_txt = self.double_stream_modulation_txt(vec)
        double_cache: list[dict[str, Tensor]] = []
        for block in self.double_blocks:
            q, k, v, pe, num_txt, mods = block._prepare_qkv(
                img, txt, pe_img, pe_txt, double_img, double_txt
            )
            q, k = apply_rope(q, k, pe)
            double_cache.append({"k": _flatten_heads(k).detach(), "v": _flatten_heads(v).detach()})
            attention = _masked_attention(q, k, v, bias)
            txt_attention, img_attention = attention[:, :num_txt], attention[:, num_txt:]
            img, txt = block._apply_residuals(img, txt, img_attention, txt_attention, mods)

        hidden = torch.cat([txt, img], dim=1)
        pe = torch.cat([pe_txt, pe_img], dim=2)
        single_mod = self.single_stream_modulation(vec)[0]
        single_cache: list[dict[str, Tensor]] = []
        for block in self.single_blocks:
            q, k, v, mlp, gate = block._qkv(hidden, single_mod)
            q, k = apply_rope(q, k, pe)
            single_cache.append({"k": _flatten_heads(k).detach(), "v": _flatten_heads(v).detach()})
            attention = _masked_attention(q, k, v, bias)
            hidden = block._out(hidden, attention, mlp, gate)
        return FrozenFluxKVCache(
            double=tuple(double_cache),
            single=tuple(single_cache),
            key_mask=key_mask.detach(),
            prefix_length=key_mask.shape[1],
            compute_dtype=self.cache_compute_dtype(),
        )

    def _expert_query_pe(self, *, batch: int, device: torch.device, dtype: torch.dtype, segment_id: int) -> Tensor:
        ids = self._robot_ids(
            batch_size=batch,
            length=1,
            segment_id=segment_id,
            device=device,
            dtype=dtype,
        )
        return self.pe_embedder(ids)

    def predict_value(
        self,
        *,
        context: Tensor,
        context_ids: Tensor,
        current_latents: Tensor,
        current_ids: Tensor,
        state: Tensor,
        context_mask: Tensor | None = None,
        expert: nn.Module | None = None,
        cache: FrozenFluxKVCache | None = None,
    ) -> Tensor:
        cache = cache if cache is not None else self.prefill_condition_cache(
            context=context,
            context_ids=context_ids,
            current_latents=current_latents,
            current_ids=current_ids,
            state=state,
            context_mask=context_mask,
        )
        selected = self.value_expert if expert is None else expert
        query_pe = self._expert_query_pe(
            batch=context.shape[0], device=context.device, dtype=current_ids.dtype, segment_id=10
        )
        return selected(cache, query_pe=query_pe)

    def predict_q(
        self,
        *,
        context: Tensor,
        context_ids: Tensor,
        current_latents: Tensor,
        current_ids: Tensor,
        state: Tensor,
        clean_action: Tensor,
        context_mask: Tensor | None = None,
        cache: FrozenFluxKVCache | None = None,
    ) -> Tensor:
        cache = cache if cache is not None else self.prefill_condition_cache(
            context=context,
            context_ids=context_ids,
            current_latents=current_latents,
            current_ids=current_ids,
            state=state,
            context_mask=context_mask,
        )
        return self.predict_q_cached(
            cache, clean_action,
            batch_indices=torch.arange(context.shape[0], device=context.device),
        )
