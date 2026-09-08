from types import SimpleNamespace

import torch
from torch import nn

from robonana.training.visualization import (
    unpack_flux2_tokens,
)


class FakeVAE(nn.Module):
    def __init__(self, packed_channels):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.bn = nn.BatchNorm2d(packed_channels, affine=False)
        self.config = SimpleNamespace(batch_norm_eps=1e-5)


def _patchify(raw):
    batch, channels, height, width = raw.shape
    return (
        raw.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, channels * 4, height // 2, width // 2)
        .flatten(2)
        .transpose(1, 2)
    )


def test_unpack_flux_tokens_inverts_patch_layout():
    raw = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
    tokens = _patchify(raw)
    vae = FakeVAE(tokens.shape[-1])
    vae.bn.running_mean.zero_()
    vae.bn.running_var.fill_(1.0 - vae.config.batch_norm_eps)
    restored = unpack_flux2_tokens(tokens, vae, grid_height=2, grid_width=3)
    torch.testing.assert_close(restored, raw)
