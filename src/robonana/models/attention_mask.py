"""FACT-style attention layout for the shared FLUX.2 backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WorldBlockMap:
    """One isolated ``[H | S | V | I | D]`` horizon-query block."""

    horizon: slice
    future_state: slice
    value: slice
    future_image: slice
    future_dino: slice


@dataclass(frozen=True)
class SegmentMap:
    """Slices for the shared prefix followed by one or more world blocks.

    A single block preserves the training layout exactly::

        [language | state | ref | A | G | H | S | V | I | D]

    Packed inference appends more mutually isolated ``[H | S | V | I | D]``
    blocks without duplicating the clean prefix or action track.
    """

    language: slice
    state: slice
    ref_image: slice
    pred_action: slice
    gt_action: slice
    world_blocks: tuple[WorldBlockMap, ...]
    total_length: int

    @property
    def clean_condition(self) -> slice:
        return slice(self.language.start, self.ref_image.stop)

    def _single_block_slice(self, name: str) -> slice:
        if len(self.world_blocks) != 1:
            raise AttributeError(
                f"{name} is ambiguous for {len(self.world_blocks)} packed world blocks; "
                "iterate over world_blocks instead"
            )
        return getattr(self.world_blocks[0], name)

    @property
    def horizon(self) -> slice:
        return self._single_block_slice("horizon")

    @property
    def future_state(self) -> slice:
        return self._single_block_slice("future_state")

    @property
    def value(self) -> slice:
        return self._single_block_slice("value")

    @property
    def future_image(self) -> slice:
        return self._single_block_slice("future_image")

    @property
    def future_dino(self) -> slice:
        return self._single_block_slice("future_dino")

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
        return cls.from_block_lengths(
            language=language,
            state=state,
            ref_image=ref_image,
            pred_action=pred_action,
            gt_action=gt_action,
            block_count=1,
            horizon=horizon,
            future_state=future_state,
            value=value,
            future_image=future_image,
            future_dino=future_dino,
        )

    @classmethod
    def from_block_lengths(
        cls,
        *,
        language: int,
        state: int,
        ref_image: int,
        pred_action: int,
        gt_action: int,
        block_count: int,
        horizon: int,
        future_state: int,
        value: int,
        future_image: int,
        future_dino: int = 0,
    ) -> "SegmentMap":
        prefix_lengths = (language, state, ref_image, pred_action, gt_action)
        block_lengths = (horizon, future_state, value, future_image, future_dino)
        lengths = (*prefix_lengths, *block_lengths)
        if any(length < 0 for length in lengths):
            raise ValueError(f"segment lengths must be non-negative, got {lengths}")
        if block_count <= 0:
            raise ValueError("block_count must be positive")

        prefix_slices: list[slice] = []
        start = 0
        for length in prefix_lengths:
            prefix_slices.append(slice(start, start + length))
            start += length
        world_blocks = []
        for _ in range(block_count):
            block_slices = []
            for length in block_lengths:
                block_slices.append(slice(start, start + length))
                start += length
            world_blocks.append(WorldBlockMap(*block_slices))
        return cls(*prefix_slices, tuple(world_blocks), total_length=start)


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
    pred_action_bidirectional: bool = False,
    context_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build additive attention bias with ``0`` allowed and ``-inf`` blocked.

    A is an isolated diffusion sink. New runs use bidirectional attention inside
    A to denoise the action chunk jointly; the default remains causal so legacy
    checkpoints without explicit mask metadata retain their original semantics.
    G is always causal. Each H/S/V/I/D block can read only the first
    ``idx_h`` full-clean G tokens and its own within-block prefix. Packed world
    blocks cannot read one another. G and all future targets cannot read A.
    """

    if not dtype.is_floating_point:
        raise TypeError(f"attention bias requires a floating dtype, got {dtype}")
    if horizon_idx.ndim == 1:
        horizon_idx = horizon_idx[:, None]
    expected_horizons = (batch_size, len(segments.world_blocks))
    if horizon_idx.ndim != 2 or tuple(horizon_idx.shape) != expected_horizons:
        raise ValueError(
            f"horizon_idx must have shape {(batch_size,)} for one block or "
            f"{expected_horizons}, got {tuple(horizon_idx.shape)}"
        )

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
    _allow(allowed, c, c)
    _allow(allowed, a, c)
    if pred_action_bidirectional:
        _allow(allowed, a, a)
    else:
        _allow_causal(allowed, a)
    _allow(allowed, g, c)
    _allow_causal(allowed, g)

    action_positions = torch.arange(gt_action_length, device=device)[None, :]
    for block_index, block in enumerate(segments.world_blocks):
        visible_gt = action_positions < horizon_idx[:, block_index, None]
        queries = (
            block.horizon,
            block.future_state,
            block.value,
            block.future_image,
            block.future_dino,
        )
        for query in queries:
            allowed[:, query, g] = visible_gt[:, None, :]

        h = block.horizon
        s = block.future_state
        v = block.value
        i = block.future_image
        d = block.future_dino
        _allow(allowed, h, c, h)
        _allow(allowed, s, c, h, s)
        _allow(allowed, v, c, h, s, v)
        _allow(allowed, i, c, h, s, v, i)
        # DINO is a trailing training-only auxiliary sink inside its block.
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
