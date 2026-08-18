import sys
from types import SimpleNamespace

import torch
from torch import nn

from robonana.training.visualization import (
    flow_prediction_to_x0,
    log_pixel_eval,
    should_log_pixel_eval,
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


def test_flow_prediction_to_x0_and_interval():
    clean = torch.tensor([[[1.0, 2.0]]])
    noise = torch.tensor([[[5.0, 8.0]]])
    timestep = torch.tensor([0.25])
    noisy = clean * 0.75 + noise * 0.25
    prediction = noise - clean
    torch.testing.assert_close(flow_prediction_to_x0(noisy, prediction, timestep), clean)
    assert not should_log_pixel_eval(1, 200)
    assert should_log_pixel_eval(200, 200)
    assert not should_log_pixel_eval(199, 200)


def test_pixel_eval_stages_images_for_same_step_commit(monkeypatch):
    calls = []

    class FakeImage:
        def __init__(self, tensor, caption):
            self.tensor = tensor
            self.caption = caption

    class FakeTracker:
        def log(self, payload, **kwargs):
            calls.append((payload, kwargs))

    class FakeAccelerator:
        is_main_process = True

        @staticmethod
        def get_tracker(name, unwrap):
            assert (name, unwrap) == ("wandb", True)
            return FakeTracker()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage))
    image = torch.zeros(2, 3, 4, 5)
    future_images = torch.zeros(2, 3, 3, 4, 5)
    log_pixel_eval(
        accelerator=FakeAccelerator(),
        step=200,
        current=image,
        targets=future_images,
        predictions=future_images,
        horizons=torch.tensor([12, 24, 48]),
    )
    assert calls[0][1] == {"step": 200, "commit": False}
    assert calls[0][0]["eval/fixed_horizon_grid"].tensor.shape == (3, 8, 35)
    assert calls[0][0]["eval/fixed_horizons"] == "12,24,48"
    assert calls[0][0]["eval/num_current_frames"] == 2
