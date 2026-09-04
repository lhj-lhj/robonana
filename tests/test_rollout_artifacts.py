import pytest
import torch

from robonana.inference.rollout_artifacts import (
    RECORDED_OVERLAY_HEIGHT,
    RECORDED_VIDEO_SIZE,
    annotate_recorded_frame,
    annotate_rollout_frame,
    decoded_frame_to_uint8,
    recorded_frame_chunk_horizon,
    split_robotwin_composite,
)
from world_action_model.image_layouts import (
    ROBOTWIN_VIEW_KEYS,
    build_robotwin_ref_tensor,
    build_robotwin_three_view_tensor,
)


def test_decoded_frame_roundtrip_and_composite_split():
    high = torch.full((3, 192, 256), 32, dtype=torch.uint8)
    left = torch.full((3, 96, 128), 96, dtype=torch.uint8)
    right = torch.full((3, 96, 128), 224, dtype=torch.uint8)
    composite = build_robotwin_three_view_tensor(
        dict(zip(ROBOTWIN_VIEW_KEYS, (high, left, right), strict=True)),
        main_dst_size=(256, 192),
    )
    decoded = composite.float().div(127.5).sub(1.0)

    recovered = decoded_frame_to_uint8(decoded)
    split = split_robotwin_composite(recovered)

    torch.testing.assert_close(split[ROBOTWIN_VIEW_KEYS[0]], high)
    torch.testing.assert_close(split[ROBOTWIN_VIEW_KEYS[1]], left)
    torch.testing.assert_close(split[ROBOTWIN_VIEW_KEYS[2]], right)


def test_return_annotation_does_not_modify_feedback_frame():
    frame = torch.zeros(3, 192, 384, dtype=torch.uint8)
    original = frame.clone()

    annotated = annotate_rollout_frame(
        frame,
        trajectory_index=2,
        rollout_index=3,
        rollout_count=5,
        horizon=48,
        reward=-3.0,
        q=-7.0,
    )

    assert annotated.size == (384, 192)
    torch.testing.assert_close(frame, original)


def test_composite_split_rejects_wrong_layout():
    with pytest.raises(ValueError, match="must have shape"):
        split_robotwin_composite(torch.zeros(3, 192, 383, dtype=torch.uint8))


@pytest.mark.parametrize(
    ("frame_index", "expected"),
    [(0, (None, 0)), (1, (0, 1)), (48, (0, 48)), (49, (1, 1)), (96, (1, 48))],
)
def test_recorded_frame_chunk_horizon(frame_index, expected):
    assert recorded_frame_chunk_horizon(frame_index, action_chunk=48) == expected


@pytest.mark.parametrize(("height", "width"), [(480, 640), (240, 320)])
def test_model_input_layout_normalizes_recorded_source_resolution(height, width):
    images = {
        key: torch.zeros(3, height, width, dtype=torch.uint8)
        for key in ROBOTWIN_VIEW_KEYS
    }

    composite = build_robotwin_ref_tensor(images, main_dst_size=(256, 192))

    assert composite.shape == (3, 192, 384)


@pytest.mark.parametrize(("height", "width"), [(480, 640), (240, 320)])
def test_recorded_annotation_normalizes_video_and_overlay_geometry(height, width):
    frame = torch.full((3, height, width), 255, dtype=torch.uint8)
    original = frame.clone()
    annotated = annotate_recorded_frame(
        frame,
        group="collected_pre_5of50",
        episode_index=7,
        frame_index=49,
        action_chunk=48,
        reward=-1.0,
        q=-12.5,
    )

    assert annotated.size == RECORDED_VIDEO_SIZE
    assert annotated.getpixel((RECORDED_VIDEO_SIZE[0] - 1, RECORDED_OVERLAY_HEIGHT - 1)) == (
        0,
        0,
        0,
    )
    assert annotated.getpixel((RECORDED_VIDEO_SIZE[0] - 1, RECORDED_OVERLAY_HEIGHT)) == (
        255,
        255,
        255,
    )
    torch.testing.assert_close(frame, original)


def test_recorded_annotation_uses_identical_font_and_bar_for_both_source_sizes():
    annotated = [
        annotate_recorded_frame(
            torch.full((3, height, width), 255, dtype=torch.uint8),
            group="expert_clean",
            episode_index=7,
            frame_index=49,
            action_chunk=48,
            reward=-1.0,
            q=-12.5,
        )
        for height, width in ((480, 640), (240, 320))
    ]

    assert annotated[0].tobytes() == annotated[1].tobytes()
