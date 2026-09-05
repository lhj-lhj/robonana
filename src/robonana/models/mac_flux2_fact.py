"""Current fixed-48 MAC actor/world model with separate MoT scalar experts."""

from __future__ import annotations

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
)


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

        # Remove every variable-horizon/flow-Q module.  Only the action/state
        # projections survive the 120k migration; all heads below are new.
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
        critic_kind: str | None = None,
    ) -> Flux2FACTOutput | Tensor:
        if critic_kind is not None:
            if critic_kind == "both":
                return (
                    self.predict_value(
                        context=context,
                        context_ids=context_ids,
                        current_latents=current_latents,
                        current_ids=current_ids,
                        state=state,
                        context_mask=context_mask,
                    ),
                    self.predict_q(
                        context=context,
                        context_ids=context_ids,
                        current_latents=current_latents,
                        current_ids=current_ids,
                        state=state,
                        clean_action=gt_action_cond,
                        context_mask=context_mask,
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
        horizon_idx = horizon_idx.reshape(-1)
        if tuple(horizon_idx.shape) != (batch,) or not bool(
            torch.all(horizon_idx == self.chunk_horizon)
        ):
            raise ValueError("mac_mot_v2 actor/world forward requires horizon_idx=48")
        del horizon_idx, noisy_reward, noisy_q
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
        for block in self.single_blocks:
            if self.gradient_checkpointing and self.training:
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
    ) -> Tensor:
        cache = self.prefill_critic_cache(
            context=context,
            context_ids=context_ids,
            current_latents=current_latents,
            current_ids=current_ids,
            state=state,
            clean_action=None,
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
    ) -> Tensor:
        cache = self.prefill_critic_cache(
            context=context,
            context_ids=context_ids,
            current_latents=current_latents,
            current_ids=current_ids,
            state=state,
            clean_action=clean_action,
            context_mask=context_mask,
        )
        query_pe = self._expert_query_pe(
            batch=context.shape[0], device=context.device, dtype=current_ids.dtype, segment_id=11
        )
        return self.q_expert(cache, query_pe=query_pe)
