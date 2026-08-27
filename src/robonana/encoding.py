"""Shared online/offline FLUX.2 image and Qwen3 encoders."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
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


@torch.inference_mode()
def encode_flux2_image_tokens(vae, images: Tensor) -> Tensor:
    """Encode normalized NCHW images into FLUX.2 image tokens."""

    raw_latents = vae.encode(images).latent_dist.mode()
    packed = patchify_and_normalize(vae, raw_latents)
    return packed.flatten(2).transpose(1, 2).contiguous()


def pixel_unshuffle_dino_patches(
    features: Tensor,
    *,
    grid_height: int = 14,
    grid_width: int = 14,
    factor: int = 2,
) -> Tensor:
    """Losslessly fold spatial DINO patches into channels."""

    batch, tokens, channels = features.shape
    if tokens != grid_height * grid_width:
        raise ValueError(
            f"DINO features must contain {grid_height * grid_width} patches, got {tokens}"
        )
    if grid_height % factor or grid_width % factor:
        raise ValueError("pixel-unshuffle factor must divide both DINO grid dimensions")
    grid = features.reshape(batch, grid_height, grid_width, channels).permute(0, 3, 1, 2)
    folded = F.pixel_unshuffle(grid, factor)
    return folded.permute(0, 2, 3, 1).reshape(
        batch,
        (grid_height // factor) * (grid_width // factor),
        channels * factor * factor,
    )


class DinoV3FeatureEncoder(nn.Module):
    """Thin frozen timm DINOv3 ViT-B/16 adapter for online targets."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("DINO training requires timm; install robonana[train]") from error
        self.model_name = str(model_name)
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self.model.eval().requires_grad_(False).to(device=torch.device(device), dtype=dtype)
        self.num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        self.embed_dim = int(getattr(self.model, "embed_dim", 0))
        if self.embed_dim != 768:
            raise RuntimeError(f"DINOv3 ViT-B/16 must have embed_dim=768, got {self.embed_dim}")
        self.register_buffer(
            "image_mean",
            torch.tensor(
                (0.485, 0.456, 0.406), device=torch.device(device), dtype=dtype
            ).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(
                (0.229, 0.224, 0.225), device=torch.device(device), dtype=dtype
            ).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def _prepare_images(self, images: Tensor, *, allow_uint8: bool) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"DINO images must be [N,3,H,W], got {tuple(images.shape)}")
        if images.dtype == torch.uint8 and allow_uint8:
            images = images.to(dtype=torch.float32).div_(255.0)
        elif not torch.is_floating_point(images):
            raise TypeError("DINO images must be floating point in [0, 1]")
        if images.numel() and (images.amin().item() < 0.0 or images.amax().item() > 1.0):
            raise ValueError("DINO images must be normalized to [0, 1]")
        model_parameter = next(self.model.parameters())
        images = images.to(device=model_parameter.device, dtype=model_parameter.dtype)
        images = F.interpolate(
            images,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return (images - self.image_mean) / self.image_std

    def _encode_prepared(self, images: Tensor) -> Tensor:
        output = self.model.forward_features(images)
        if isinstance(output, Mapping):
            try:
                patches = output["x_norm_patchtokens"]
            except KeyError as error:
                raise RuntimeError(
                    f"Unsupported DINO forward_features keys: {sorted(output)}"
                ) from error
        else:
            patches = output[:, self.num_prefix_tokens :]
        if tuple(patches.shape[1:]) != (196, 768):
            raise RuntimeError(f"Expected native DINO patches [N,196,768], got {tuple(patches.shape)}")
        return pixel_unshuffle_dino_patches(patches, factor=2)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """Encode RGB ``[N,3,H,W]`` images in ``[0,1]`` to ``[N,49,3072]``."""

        return self._encode_prepared(self._prepare_images(images, allow_uint8=False))

    @torch.no_grad()
    def encode_views(
        self,
        images_by_view: Mapping[str, Tensor],
        *,
        view_keys: tuple[str, ...],
        inference_batch_size: int,
    ) -> Tensor:
        """Encode ordered per-camera RGB batches into ``[B,V*49,3072]``.

        Each camera may have a different source resolution. Inputs may be
        ``uint8`` in ``[0,255]`` or floating point in ``[0,1]``.
        """

        if inference_batch_size <= 0:
            raise ValueError("DINO inference_batch_size must be positive")
        if not view_keys:
            raise ValueError("DINO view_keys must be non-empty")
        missing = [key for key in view_keys if key not in images_by_view]
        if missing:
            raise KeyError(f"DINO input is missing camera views: {missing}")

        prepared = []
        batch_size = None
        for key in view_keys:
            images = images_by_view[key]
            if batch_size is None:
                batch_size = images.shape[0]
            elif images.shape[0] != batch_size:
                raise ValueError("all DINO camera views must have the same batch size")
            prepared.append(self._prepare_images(images, allow_uint8=True))
        all_images = torch.cat(prepared, dim=0)
        features = torch.cat(
            [
                self._encode_prepared(all_images[start : start + inference_batch_size])
                for start in range(0, all_images.shape[0], inference_batch_size)
            ],
            dim=0,
        )
        assert batch_size is not None
        return features.reshape(len(view_keys), batch_size, 49, 3072).permute(1, 0, 2, 3).reshape(
            batch_size,
            len(view_keys) * 49,
            3072,
        )
