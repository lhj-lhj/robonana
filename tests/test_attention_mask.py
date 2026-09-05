import torch

from robonana.models.attention_mask import (
    MacSegmentMap,
    SegmentMap,
    build_attention_bias,
    build_mac_attention_bias,
    build_mac_critic_prefix_bias,
)


def test_mac_mask_contains_only_actor_world_cascade():
    seg = MacSegmentMap.from_lengths(
        language=2,
        state=1,
        ref_image=2,
        pred_action=4,
        clean_action=4,
        reward=1,
        success=1,
        future_state=1,
        future_image=3,
    )
    bias = build_mac_attention_bias(
        seg,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
    )[0, 0]

    context = seg.clean_condition
    assert torch.isfinite(bias[seg.reward, seg.clean_action]).all()
    assert torch.isfinite(bias[seg.success, seg.reward]).all()
    assert torch.isneginf(bias[seg.success, seg.future_state]).all()
    assert torch.isfinite(bias[seg.future_state, seg.reward]).all()
    assert torch.isfinite(bias[seg.future_state, seg.success]).all()
    assert torch.isfinite(bias[seg.future_image, seg.future_state]).all()
    for world in (seg.reward, seg.success, seg.future_state, seg.future_image):
        assert torch.isneginf(bias[world, seg.pred_action]).all()


def test_mac_value_prefix_has_no_action_and_q_prefix_has_full_action():
    value_bias, value_keys = build_mac_critic_prefix_bias(
        language_length=2,
        state_length=1,
        image_length=2,
        action_length=0,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
    )
    q_bias, q_keys = build_mac_critic_prefix_bias(
        language_length=2,
        state_length=1,
        image_length=2,
        action_length=48,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
    )
    assert value_bias.shape[-1] == value_keys.shape[-1] == 5
    assert q_bias.shape[-1] == q_keys.shape[-1] == 53
    assert torch.isfinite(q_bias).all()


def test_pred_action_is_bidirectional_gt_action_is_causal_and_targets_see_prefix():
    seg = SegmentMap.from_lengths(
        language=2,
        state=1,
        ref_image=2,
        pred_action=4,
        gt_action=4,
        horizon=1,
        future_state=1,
        reward=1,
        success=1,
        q=1,
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
        reward=1,
        success=1,
        q=1,
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
        reward=1,
        success=1,
        q=1,
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
        reward=1,
        success=1,
        q=1,
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
        reward=1,
        success=1,
        q=1,
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

    for query in (first.horizon, first.future_state, first.reward, first.success, first.q):
        assert torch.isfinite(bias[query, seg.gt_action.start]).all()
        assert torch.isneginf(bias[query, seg.gt_action.start + 1 : seg.gt_action.stop]).all()
    for query in (second.horizon, second.future_state, second.reward, second.success, second.q):
        assert torch.isfinite(bias[query, seg.gt_action.start : seg.gt_action.start + 3]).all()
        assert torch.isneginf(bias[query, seg.gt_action.start + 3 : seg.gt_action.stop]).all()

    first_span = slice(first.horizon.start, first.future_dino.stop)
    second_span = slice(second.horizon.start, second.future_dino.stop)
    assert torch.isneginf(bias[first_span, second_span]).all()
    assert torch.isneginf(bias[second_span, first_span]).all()


def test_reward_q_follow_world_block_order_and_never_read_pred_action():
    seg = SegmentMap.from_lengths(
        language=1,
        state=1,
        ref_image=1,
        pred_action=3,
        gt_action=3,
        horizon=1,
        future_state=1,
        reward=1,
        success=1,
        q=1,
        future_image=1,
        future_dino=1,
    )
    bias = build_attention_bias(
        seg,
        batch_size=1,
        dtype=torch.float32,
        device="cpu",
        horizon_idx=torch.tensor([2]),
        pred_action_bidirectional=True,
    )[0, 0]

    assert torch.isfinite(bias[seg.reward, seg.future_state]).all()
    assert torch.isneginf(bias[seg.reward, seg.success]).all()
    assert torch.isfinite(bias[seg.success, seg.reward]).all()
    assert torch.isneginf(bias[seg.success, seg.q]).all()
    assert torch.isfinite(bias[seg.q, seg.success]).all()
    assert torch.isneginf(bias[seg.q, seg.future_image]).all()
    for query in (seg.reward, seg.success, seg.q):
        assert torch.isneginf(bias[query, seg.pred_action]).all()
        assert torch.isfinite(bias[query, seg.gt_action.start : seg.gt_action.start + 2]).all()
        assert torch.isneginf(bias[query, seg.gt_action.start + 2 : seg.gt_action.stop]).all()
