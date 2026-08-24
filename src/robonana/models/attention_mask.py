"""FACT-style attention layout for the shared FLUX.2 backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SegmentMap:
    """Slices for ``[language | state | ref | A | G | H | S | V | I | D]``."""

    language: slice
    state: slice
    ref_image: slice
    pred_action: slice
    gt_action: slice
    horizon: slice
    future_state: slice
    value: slice
    future_image: slice
    future_dino: slice
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
        gt_action: int,
        horizon: int,
        future_state: int,
        value: int,
        future_image: int,
        future_dino: int = 0,
    ) -> "SegmentMap":
        lengths = (
            language,
            state,
            ref_image,
            pred_action,
            gt_action,
            horizon,
            future_state,
            value,
            future_image,
            future_dino,
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


def build_attention_bias(
    segments: SegmentMap,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
    horizon_idx: torch.Tensor,
    context_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build additive attention bias with ``0`` allowed and ``-inf`` blocked.

    Both action tracks are causal: action token ``t`` cannot read ``t + 1``.
    A is a sink; G and all future targets cannot read A.  For each sample,
    H/S/V/I/D can read only the first ``idx_h`` full-clean G tokens.
    """

    if not dtype.is_floating_point:
        raise TypeError(f"attention bias requires a floating dtype, got {dtype}")
    if horizon_idx.ndim != 1 or tuple(horizon_idx.shape) != (batch_size,):
        raise ValueError(f"horizon_idx must have shape {(batch_size,)}, got {tuple(horizon_idx.shape)}")

    gt_action_length = segments.gt_action.stop - segments.gt_action.start
    horizon_idx = horizon_idx.to(device=device, dtype=torch.long)
    if torch.any(horizon_idx < 1) or torch.any(horizon_idx > gt_action_length):
        raise ValueError(
            f"horizon_idx must lie in [1, {gt_action_length}] so future targets have a valid G prefix"
        )

    n = segments.total_length
    allowed = torch.zeros(batch_size, n, n, dtype=torch.bool, device=device)
    c = segments.clean_condition
    a = segments.pred_action
    g = segments.gt_action
    h = segments.horizon
    s = segments.future_state
    v = segments.value
    i = segments.future_image
    d = segments.future_dino

    _allow(allowed, c, c)
    _allow(allowed, a, c)
    _allow_causal(allowed, a)
    _allow(allowed, g, c)
    _allow_causal(allowed, g)

    visible_gt = (
        torch.arange(gt_action_length, device=device)[None, :]
        < horizon_idx[:, None]
    )
    for query in (h, s, v, i, d):
        allowed[:, query, g] = visible_gt[:, None, :]

    _allow(allowed, h, c, h)
    _allow(allowed, s, c, h, s)
    _allow(allowed, v, c, h, s, v)
    _allow(allowed, i, c, h, s, v, i)
    # DINO is a trailing training-only auxiliary sink. It can use the complete
    # world-model path, while no earlier token can depend on DINO features.
    _allow(allowed, d, c, h, s, v, i, d)

    if context_mask is not None:
        expected = (batch_size, segments.language.stop - segments.language.start)
        if tuple(context_mask.shape) != expected:
            raise ValueError(f"context_mask must have shape {expected}, got {tuple(context_mask.shape)}")
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        valid_keys = torch.ones(batch_size, n, dtype=torch.bool, device=device)
        valid_keys[:, segments.language] = context_mask
        allowed &= valid_keys[:, None, :]

        # Avoid all-masked rows for padded language queries. Their outputs are
        # ignored, but SDPA still needs one finite key to avoid NaNs.
        for batch_index in range(batch_size):
            padded = torch.where(~context_mask[batch_index])[0] + segments.language.start
            allowed[batch_index, padded, :] = False
            allowed[batch_index, padded, padded] = True

    bias = torch.zeros(batch_size, 1, n, n, dtype=dtype, device=device)
    return bias.masked_fill(~allowed[:, None], float("-inf"))
