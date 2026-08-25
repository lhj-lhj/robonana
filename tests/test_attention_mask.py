import torch

from robonana.models.attention_mask import SegmentMap, build_attention_bias


def test_pred_action_is_bidirectional_gt_action_is_causal_and_targets_see_prefix():
    seg = SegmentMap.from_lengths(
        language=2,
        state=1,
        ref_image=2,
        pred_action=4,
        gt_action=4,
        horizon=1,
        future_state=1,
        value=1,
        future_image=3,
    )
    bias = build_attention_bias(
        seg,
        batch_size=2,
        dtype=torch.float32,
        device="cpu",
        horizon_idx=torch.tensor([1, 3]),
        pred_action_bidirectional=True,
    )[:, 0]

    for batch_bias in bias:
        assert torch.isfinite(batch_bias[seg.pred_action, seg.pred_action]).all()
        gt_block = batch_bias[seg.gt_action, seg.gt_action]
        causal = torch.ones_like(gt_block, dtype=torch.bool).tril()
        assert torch.isfinite(gt_block[causal]).all()
        assert torch.isneginf(gt_block[~causal]).all()
        assert torch.isneginf(batch_bias[seg.pred_action, seg.gt_action]).all()
        assert torch.isneginf(batch_bias[seg.gt_action, seg.pred_action]).all()
        assert torch.isneginf(batch_bias[seg.future_image, seg.pred_action]).all()

    for batch_index, horizon in enumerate((1, 3)):
        gt_visibility = bias[batch_index, seg.future_image, seg.gt_action]
        assert torch.isfinite(gt_visibility[:, :horizon]).all()
        assert torch.isneginf(gt_visibility[:, horizon:]).all()


def test_legacy_mask_default_keeps_both_action_tracks_causal():
    seg = SegmentMap.from_lengths(
        language=1,
        state=1,
        ref_image=1,
        pred_action=3,
        gt_action=3,
        horizon=1,
        future_state=1,
        value=1,
        future_image=1,
    )
    bias = build_attention_bias(
        seg,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
        horizon_idx=torch.tensor([2]),
    )[0, 0]

    for segment in (seg.pred_action, seg.gt_action):
        block = bias[segment, segment]
        causal = torch.ones_like(block, dtype=torch.bool).tril()
        assert torch.isfinite(block[causal]).all()
        assert torch.isneginf(block[~causal]).all()


def test_padded_context_keys_are_blocked_without_all_masked_rows():
    seg = SegmentMap.from_lengths(
        language=3,
        state=1,
        ref_image=1,
        pred_action=1,
        gt_action=1,
        horizon=1,
        future_state=1,
        value=1,
        future_image=1,
    )
    context_mask = torch.tensor([[True, True, False]])
    bias = build_attention_bias(
        seg,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
        horizon_idx=torch.tensor([1]),
        context_mask=context_mask,
    )[0, 0]
    padded = 2
    assert not torch.isneginf(bias[:, padded]).all().item()
    assert torch.isfinite(bias[padded, padded])
    valid_queries = torch.arange(seg.total_length) != padded
    assert torch.isneginf(bias[valid_queries, padded]).all()


def test_dino_is_trailing_one_way_auxiliary_sink():
    seg = SegmentMap.from_lengths(
        language=2,
        state=1,
        ref_image=2,
        pred_action=2,
        gt_action=2,
        horizon=1,
        future_state=1,
        value=1,
        future_image=3,
        future_dino=4,
    )
    bias = build_attention_bias(
        seg,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
        horizon_idx=torch.tensor([1]),
    )[0, 0]

    assert torch.isfinite(bias[seg.future_dino, seg.future_image]).all()
    assert torch.isfinite(bias[seg.future_dino, seg.future_dino]).all()
    assert torch.isfinite(bias[seg.future_dino, seg.gt_action.start]).all()
    assert torch.isneginf(
        bias[seg.future_dino, seg.gt_action.start + 1 : seg.gt_action.stop]
    ).all()
    assert torch.isneginf(bias[seg.future_image, seg.future_dino]).all()
    assert torch.isneginf(bias[seg.pred_action, seg.future_dino]).all()


def test_packed_horizon_blocks_see_their_action_prefix_but_not_each_other():
    seg = SegmentMap.from_block_lengths(
        language=1,
        state=1,
        ref_image=2,
        pred_action=0,
        gt_action=4,
        block_count=2,
        horizon=1,
        future_state=1,
        value=1,
        future_image=0,
    )
    bias = build_attention_bias(
        seg,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
        horizon_idx=torch.tensor([[1, 3]]),
    )[0, 0]
    first, second = seg.world_blocks

    for query in (first.horizon, first.future_state, first.value):
        assert torch.isfinite(bias[query, seg.gt_action.start]).all()
        assert torch.isneginf(bias[query, seg.gt_action.start + 1 : seg.gt_action.stop]).all()
    for query in (second.horizon, second.future_state, second.value):
        assert torch.isfinite(bias[query, seg.gt_action.start : seg.gt_action.start + 3]).all()
        assert torch.isneginf(bias[query, seg.gt_action.start + 3 : seg.gt_action.stop]).all()

    first_span = slice(first.horizon.start, first.future_dino.stop)
    second_span = slice(second.horizon.start, second.future_dino.stop)
    assert torch.isneginf(bias[first_span, second_span]).all()
    assert torch.isneginf(bias[second_span, first_span]).all()
