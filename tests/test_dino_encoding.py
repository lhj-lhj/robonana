import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from robonana.encoding import DinoV3FeatureEncoder, pixel_unshuffle_dino_patches


def test_dino_pixel_unshuffle_is_lossless_space_to_channel():
    features = torch.arange(1 * 4 * 4 * 3).reshape(1, 16, 3).float()
    folded = pixel_unshuffle_dino_patches(
        features,
        grid_height=4,
        grid_width=4,
        factor=2,
    )
    assert folded.shape == (1, 4, 12)
    folded_grid = folded.reshape(1, 2, 2, 12).permute(0, 3, 1, 2)
    restored = F.pixel_shuffle(folded_grid, 2).permute(0, 2, 3, 1).reshape(1, 16, 3)
    torch.testing.assert_close(restored, features)


def test_dino_encoder_uses_patch_tokens_and_returns_lossless_fold(monkeypatch):
    class FakeDino(nn.Module):
        num_prefix_tokens = 1
        embed_dim = 768

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward_features(self, images):
            batch = images.shape[0]
            return torch.arange(197 * 768, device=images.device, dtype=images.dtype).reshape(
                1, 197, 768
            ).expand(batch, -1, -1)

    monkeypatch.setitem(
        __import__("sys").modules,
        "timm",
        SimpleNamespace(create_model=lambda *args, **kwargs: FakeDino()),
    )
    encoder = DinoV3FeatureEncoder(device="cpu")
    output = encoder(torch.zeros(2, 3, 32, 24))
    assert output.shape == (2, 49, 3072)


def test_dino_encoder_online_views_accept_uint8_and_preserve_camera_order(monkeypatch):
    class FakeDino(nn.Module):
        num_prefix_tokens = 1
        embed_dim = 768

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward_features(self, images):
            per_image = images.mean(dim=(1, 2, 3)).reshape(-1, 1, 1)
            patches = per_image.expand(-1, 196, 768)
            prefix = torch.zeros(images.shape[0], 1, 768, device=images.device, dtype=images.dtype)
            return torch.cat([prefix, patches], dim=1)

    monkeypatch.setitem(
        __import__("sys").modules,
        "timm",
        SimpleNamespace(create_model=lambda *args, **kwargs: FakeDino()),
    )
    encoder = DinoV3FeatureEncoder(device="cpu")
    output = encoder.encode_views(
        {
            "high": torch.zeros(2, 3, 32, 24, dtype=torch.uint8),
            "left": torch.full((2, 3, 16, 12), 64, dtype=torch.uint8),
            "right": torch.full((2, 3, 8, 6), 255, dtype=torch.uint8),
        },
        view_keys=("high", "left", "right"),
        inference_batch_size=2,
    )
    assert output.shape == (2, 147, 3072)
    assert not torch.is_inference(output)
    assert not torch.equal(output[:, :49], output[:, 49:98])
    assert not torch.equal(output[:, 49:98], output[:, 98:])


@pytest.mark.skipif(
    os.environ.get("ROBONANA_TEST_REAL_DINO") != "1",
    reason="opt-in Hugging Face weight download and GPU smoke",
)
def test_real_dinov3_vitb16_checkpoint_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real DINO smoke")
    encoder = DinoV3FeatureEncoder(device="cuda")
    output = encoder.encode_views(
        {
            "high": torch.zeros(1, 3, 192, 256, dtype=torch.uint8),
            "left": torch.zeros(1, 3, 96, 128, dtype=torch.uint8),
            "right": torch.zeros(1, 3, 96, 128, dtype=torch.uint8),
        },
        view_keys=("high", "left", "right"),
        inference_batch_size=3,
    )
    assert output.shape == (1, 147, 3072)
    assert torch.isfinite(output).all()
    assert not output.requires_grad
