"""Shared online/offline FLUX.2 image and Qwen3 encoders."""

from __future__ import annotations

from threading import RLock
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
            # Match the official FLUX.2 Qwen3 loader. Passing a torch dtype
            # through Transformers' newer ``dtype=`` alias mutates the config
            # before it is logged and breaks older versions whose JSON encoder
            # cannot serialize ``torch.dtype``.
            torch_dtype=None,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).eval()
        self.model.requires_grad_(False)
        self.model.to(device=device, dtype=torch.bfloat16)
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


_VAE_LOCK = RLock()


@torch.inference_mode()
def encode_flux2_image_tokens(vae, images: Tensor) -> Tensor:
    """Unique cache/live contract: FP32, one image, no TF32, BF16 roundtrip.

    VAE weights and Qwen are unchanged. Batch grouping is only an I/O concern;
    every convolution sees N=1, including cache generation and live batch=2.
    Scoped backend flags are restored; never change Qwen/FLUX global settings.
    """
    parameter = next(vae.parameters())
    if parameter.dtype != torch.float32 or vae.training:
        raise ValueError("Image pipeline requires frozen eval-mode FP32 VAE")
    if images.ndim != 4 or images.shape[0] == 0 or images.dtype != torch.float32:
        raise ValueError("VAE input must be a nonempty FP32 NCHW batch")
    outputs = []
    with _VAE_LOCK:
        precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            with torch.autocast(device_type=parameter.device.type, enabled=False), torch.backends.cudnn.flags(
                enabled=True, benchmark=False, deterministic=True, allow_tf32=False
            ):
                for image in images.split(1):
                    raw = vae.encode(image).latent_dist.mode()
                    packed = patchify_and_normalize(vae, raw)
                    tokens = packed.flatten(2).transpose(1, 2).contiguous()
                    outputs.append(tokens.to(torch.bfloat16).to(torch.float32))
        finally:
            torch.set_float32_matmul_precision(precision)
    return torch.cat(outputs)
