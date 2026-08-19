"""Shared online/offline FLUX.2 image and Qwen3 encoders."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from flux2.text_encoder import MAX_LENGTH, Qwen3Embedder


class LocalQwen3Embedder(Qwen3Embedder):
    """Use the official FLUX.2 Qwen3 forward with local component folders."""

    def __init__(self, checkpoint: str | Path, device: torch.device | str) -> None:
        nn.Module.__init__(self)
        checkpoint = Path(checkpoint)
        device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint / "text_encoder",
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint / "tokenizer",
            local_files_only=True,
        )
        self.max_length = MAX_LENGTH


def patchify_and_normalize(vae, latents: Tensor) -> Tensor:
    """Apply the FLUX.2 AE 2x2 packing and checkpoint batch normalization."""

    batch, channels, height, width = latents.shape
    if height % 2 or width % 2:
        raise ValueError(f"FLUX.2 VAE latent spatial shape must be even, got {(height, width)}")
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4).reshape(
        batch,
        channels * 4,
        height // 2,
        width // 2,
    )
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
        device=latents.device,
        dtype=latents.dtype,
    )
    std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(device=latents.device, dtype=latents.dtype)
    return (latents - mean) / std


@torch.inference_mode()
def encode_flux2_image_tokens(vae, images: Tensor) -> Tensor:
    """Encode normalized NCHW images into FLUX.2 image tokens."""

    raw_latents = vae.encode(images).latent_dist.mode()
    packed = patchify_and_normalize(vae, raw_latents)
    return packed.flatten(2).transpose(1, 2).contiguous()
