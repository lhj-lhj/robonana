"""FACT-style attention layout for the shared FLUX.2 backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MacSegmentMap:
    """Fixed-chunk actor/world layout for ``mac_mot_v2``.

    The language tokens are stored in FLUX's text stream and every other slice
    is stored in the image stream, but the slices describe their concatenated
    attention order::

        [L | S | I | A | G | R | U | S' | I']

    ``R`` is one learned query whose output head produces all 48 reward logits.
    ``U`` is the success-terminal query.  Value and Q are separate MoT-style
    experts and therefore never appear in this shared sequence.
    """

    language: slice
    state: slice
    ref_image: slice
    pred_action: slice
    clean_action: slice
    reward: slice
    success: slice
    future_state: slice
    future_image: slice
    total_length: int

    @property
    def clean_condition(self) -> slice:
        return slice(self.language.start, self.ref_image.stop)

    @classmethod
    def from_lengths(
        cls,
        *,
        language: int,
        state: int,
        ref_image: int,
        pred_action: int,
        clean_action: int,
        reward: int,
        success: int,
        future_state: int,
        future_image: int,
    ) -> "MacSegmentMap":
        lengths = (
            language,
            state,
            ref_image,
            pred_action,
            clean_action,
            reward,
            success,
            future_state,
            future_image,
        )
        if any(length < 0 for length in lengths):
            raise ValueError(f"segment lengths must be non-negative, got {lengths}")
        slices: list[slice] = []
        start = 0
        for length in lengths:
            slices.append(slice(start, start + length))
            start += length
        return cls(*slices, total_length=start)


def _allow(allowed: torch.Tensor, query: slice, *keys: slice) -> None:
    for key in keys:
        allowed[:, query, key] = True


def _allow_causal(allowed: torch.Tensor, segment: slice) -> None:
    length = segment.stop - segment.start
    if length:
        allowed[:, segment, segment] = torch.ones(
            length,
            length,
            dtype=torch.bool,
            device=allowed.device,
        ).tril()


def build_mac_attention_bias(
    segments: MacSegmentMap,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
    context_mask: torch.Tensor | None = None,
    world_conditioning: str = "fixed48",
    world_horizon: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the explicit fixed-chunk MAC dependency graph.

    The graph is intentionally stronger than an ordinary causal mask.  The
    fixed48 world model uses the cascade ``R -> U -> S' -> I'``;
    rope_prefix isolates dense R from the horizon-conditioned U/S'/I'. The
    the noisy policy track is an isolated sink.  Q and Value live outside this
    sequence and consume frozen-prefix K/V only.
    """

    if not dtype.is_floating_point:
        raise TypeError(f"attention bias requires a floating dtype, got {dtype}")
    if world_conditioning not in {"fixed48", "rope_prefix"}:
        raise ValueError("world_conditioning must be fixed48 or rope_prefix")
    if world_conditioning == "fixed48" and world_horizon is not None:
        raise ValueError("world_horizon is only accepted by rope_prefix")
    n = segments.total_length
    allowed = torch.zeros(batch_size, n, n, dtype=torch.bool, device=device)
    c = segments.clean_condition
    a = segments.pred_action
    g = segments.clean_action
    r = segments.reward
    u = segments.success
    s = segments.future_state
    i = segments.future_image

    _allow(allowed, c, c)
    _allow(allowed, a, c, a)
    # A candidate is a complete known chunk, so the clean conditioning track
    # is bidirectional and every downstream query may inspect all 48 actions.
    _allow(allowed, g, c, g)
    _allow(allowed, r, c, g, r)
    _allow(allowed, u, c, g, r, u)
    _allow(allowed, s, c, g, r, u, s)
    _allow(allowed, i, c, g, r, u, s, i)

    if world_conditioning == "rope_prefix":
        if world_horizon is None or tuple(world_horizon.shape) != (batch_size,):
            raise ValueError("rope_prefix requires world_horizon with shape [B]")
        if world_horizon.dtype not in (torch.int32, torch.int64):
            raise ValueError("world_horizon must contain integer frame offsets")
        h = world_horizon.to(device=device)
        if bool(torch.any((h < 1) | (h > 48))):
            raise ValueError("world_horizon must lie in [1,48]")
        if g.stop - g.start not in (0, 48):
            raise ValueError("rope_prefix requires an empty or 48-step clean action chunk")
        # C cannot read actions, A is an isolated sink, and G is causal.
        # Dense R sees the full chunk. U/S'/I' must never read R,
        # otherwise R would carry suffix actions across transformer layers.
        allowed[:, u.start:i.stop, r] = False
        _allow_causal(allowed, g)
        visible = torch.arange(g.stop - g.start, device=device)[None, :] < h[:, None]
        allowed[:, u.start:i.stop, g] &= visible[:, None, :]

    if context_mask is not None:
        expected = (batch_size, segments.language.stop - segments.language.start)
        if tuple(context_mask.shape) != expected:
            raise ValueError(
                f"context_mask must have shape {expected}, got {tuple(context_mask.shape)}"
            )
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        valid_keys = torch.ones(batch_size, n, dtype=torch.bool, device=device)
        valid_keys[:, segments.language] = context_mask
        allowed &= valid_keys[:, None, :]
        for batch_index in range(batch_size):
            padded = (
                torch.where(~context_mask[batch_index])[0] + segments.language.start
            )
            allowed[batch_index, padded, :] = False
            allowed[batch_index, padded, padded] = True

    bias = torch.zeros(batch_size, 1, n, n, dtype=dtype, device=device)
    return bias.masked_fill(~allowed[:, None], float("-inf"))


def build_mac_critic_prefix_bias(
    *,
    language_length: int,
    state_length: int,
    image_length: int,
    action_length: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
    context_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention mask for frozen FLUX critic prefixes.

    Value passes ``action_length=0`` and therefore caches exactly
    ``[language, state, current_image]``.  Q additionally caches the complete
    clean action chunk. C reads only C; each complete G reads C and itself.
    In particular C must never depend on G, so it can be shared across action
    candidates and agrees with the actor/world clean-condition track.
    This function returns both the square backbone bias
    and the flat valid-key mask reused by the one-query expert.
    """

    if not dtype.is_floating_point:
        raise TypeError(f"attention bias requires a floating dtype, got {dtype}")
    lengths = (language_length, state_length, image_length, action_length)
    if any(int(length) < 0 for length in lengths):
        raise ValueError(f"critic prefix lengths must be non-negative, got {lengths}")
    total = sum(lengths)
    allowed = torch.ones(batch_size, total, total, dtype=torch.bool, device=device)
    condition_length = language_length + state_length + image_length
    allowed[:, :condition_length, condition_length:] = False
    key_mask = torch.ones(batch_size, total, dtype=torch.bool, device=device)
    if context_mask is not None:
        expected = (batch_size, language_length)
        if tuple(context_mask.shape) != expected:
            raise ValueError(
                f"context_mask must have shape {expected}, got {tuple(context_mask.shape)}"
            )
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        key_mask[:, :language_length] = context_mask
        allowed &= key_mask[:, None, :]
        for batch_index in range(batch_size):
            padded = torch.where(~context_mask[batch_index])[0]
            allowed[batch_index, padded, :] = False
            allowed[batch_index, padded, padded] = True
    bias = torch.zeros(batch_size, 1, total, total, dtype=dtype, device=device)
    return bias.masked_fill(~allowed[:, None], float("-inf")), key_mask
