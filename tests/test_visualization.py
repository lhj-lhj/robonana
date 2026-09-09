from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from flux2.autoencoder import AutoEncoder

from robonana.training.visualization import (
    unpack_flux2_tokens,
    decode_flux2_tokens,
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_tokens_use_official_fp32_bn_inverse_before_decoder(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Real CUDA dtype promotion requires a GPU")
    vae = FakeVAE(4).to(device).eval()
    vae.bn_eps = vae.config.batch_norm_eps
    vae.bn.running_mean.fill_(.123456)
    vae.bn.running_var.fill_(1.234567)
    tokens = torch.tensor([[[1., 2., 3., 4.]]], device=device, dtype=torch.bfloat16)
    packed = tokens.transpose(1, 2).reshape(1, 4, 1, 1)
    # Use upstream's real implementation as the oracle, not a duplicate formula.
    expected = AutoEncoder.inv_normalize(vae, packed).reshape(1, 1, 2, 2)
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = unpack_flux2_tokens(tokens, vae, grid_height=1, grid_width=1)
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with patch.object(vae, "decode", create=True, return_value=(torch.zeros_like(expected),)) as decode:
        decode_flux2_tokens(vae, tokens, grid_height=1, grid_width=1)
    torch.testing.assert_close(decode.call_args.args[0], expected, rtol=0, atol=0)
    assert vae.bn.running_mean.dtype == vae.bn.running_var.dtype == torch.float32
