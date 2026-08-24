from types import SimpleNamespace

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
