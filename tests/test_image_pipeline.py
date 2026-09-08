"""Cache/live image parity: test actual adapters, not two copies of the math."""
import json
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from robonana import image_pipeline as pipeline
from robonana.data.rollout_writer import _png_bytes
from robonana.encoding import encode_flux2_image_tokens


def views(seed=0):
    rng = np.random.default_rng(seed)
    return {key: rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
            for key in pipeline.VIEW_KEYS}


def test_pixels_match_original_fact_and_live_float_and_batch():
    raw = views()
    expected = pipeline.fact_preprocess()._build_composite(
        {key: value[None] for key, value in raw.items()}, (256, 192), list(pipeline.VIEW_KEYS))
    assert torch.equal(expected, pipeline.build_robotwin_vae_input(raw))
    live = {key: torch.from_numpy(value).permute(2, 0, 1).float() / 255
            for key, value in raw.items()}
    assert torch.equal(expected, pipeline.build_robotwin_vae_input(live))
    batch = {key: np.stack([value, value]) for key, value in raw.items()}
    assert torch.equal(expected.repeat(2, 1, 1, 1), pipeline.build_robotwin_vae_input(batch))


def test_png_roundtrip_preserves_live_pixels():
    raw = views()
    decoded = {}
    for key, value in raw.items():
        for image in (value, value.astype(np.float32) / 255):
            with Image.open(BytesIO(_png_bytes(image))) as saved:
                assert saved.format == "PNG"
                assert np.array_equal(np.asarray(saved), value)
                decoded[key] = np.asarray(saved).copy()
    assert torch.equal(pipeline.build_robotwin_vae_input(raw), pipeline.build_robotwin_vae_input(decoded))


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.bn = torch.nn.BatchNorm2d(128)
        self.config = SimpleNamespace(batch_norm_eps=1e-5)
        self.seen = []
        self.eval()

    def encode(self, image):
        assert not torch.backends.cudnn.allow_tf32
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert torch.get_float32_matmul_precision() == "highest"
        self.seen.append(image.shape[0])
        value = image[:, :1, :2, :2].repeat(1, 32, 1, 1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: value))


def test_vae_batch_one_rounding_and_backend_restoration():
    vae = FakeVAE()
    images = torch.randn(2, 3, 4, 4)
    before = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32,
              torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    batched = encode_flux2_image_tokens(vae, images)
    solo = torch.cat([encode_flux2_image_tokens(vae, x[None]) for x in images])
    assert vae.seen == [1, 1, 1, 1]
    assert torch.equal(batched, solo)
    assert torch.equal(batched, batched.bfloat16().float())
    assert before == (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32,
                      torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    with pytest.raises(ValueError):
        encode_flux2_image_tokens(vae.train(), images)


def test_cache_contract_and_training_pools_fail_closed(tmp_path, monkeypatch):
    contract = {"version": pipeline.IMAGE_PIPELINE_VERSION, "vae_sha256": "test-only"}
    monkeypatch.setattr(pipeline, "image_contract", lambda _: contract)
    pipeline.write_image_contract(tmp_path, tmp_path)
    pipeline.write_image_contract(tmp_path, tmp_path)  # idempotent
    output = tmp_path / "flux_cache/latents_v2/episode_000000.pt"
    pipeline.save_image_cache(torch.zeros(2, 288, 128), output)
    assert pipeline.valid_image_cache(output, 2)
    assert not pipeline.valid_image_cache(output, 3)
    dataset = SimpleNamespace(_ensure_index=lambda: None,
                              records=[SimpleNamespace(task_dir=tmp_path)])
    pipeline.validate_training_image_contracts(SimpleNamespace(datasets=[dataset]), tmp_path)
    proof = output.with_suffix(".json")
    proof.unlink()
    assert not pipeline.valid_image_cache(output)
    (output.parent / "_contract.json").write_text(json.dumps({**contract, "vae_sha256": "wrong"}))
    with pytest.raises(RuntimeError, match="mismatch"):
        pipeline.validate_training_image_contracts(dataset, tmp_path)
    with pytest.raises(RuntimeError):
        pipeline.write_image_contract(tmp_path, tmp_path)
    with pytest.raises(RuntimeError, match="rebuild"):
        pipeline.require_image_contract(tmp_path / "legacy")
