"""One image-input contract for LeRobot, HDF5 replay and live inference.

Reuse FACT's ORIGINAL LeRobot pixel transform instead of maintaining another
resize implementation: scripts/compute_vae_latents.py::_build_composite and
::_per_view_transform in https://github.com/InternRobotics/FACT .
Only source decoding differs (MP4 / encoded RGB / live RGB). Model input math,
single-image VAE execution and cache rounding are shared by every consumer.
"""
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

IMAGE_PIPELINE_VERSION = "robotwin_fact_flux_image_v2"
MAIN_VIEW_SIZE = (256, 192)
VIEW_KEYS = ("observation.images.cam_high", "observation.images.cam_left_wrist",
             "observation.images.cam_right_wrist")


@lru_cache(maxsize=1)
def fact_preprocess():
    import world_action_model
    path = Path(world_action_model.__file__).resolve().parents[1] / "scripts/compute_vae_latents.py"
    spec = importlib.util.spec_from_file_location("robonana_fact_pixel_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FACT pixel helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rgb_thwc(value):
    value = torch.as_tensor(value).detach().cpu()
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4:
        raise ValueError("RGB must be CHW/HWC or NCHW/NHWC")
    if value.shape[-1] != 3 and value.shape[1] == 3:
        value = value.permute(0, 2, 3, 1)
    if value.shape[-1] != 3:
        raise ValueError("RGB needs three channels")
    if value.dtype != torch.uint8:
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("RGB must be uint8 or finite float in [0,1]")
        if value.numel() and (value.min() < 0 or value.max() > 1):
            raise ValueError("floating RGB must be in [0,1]")
        # FACT's live client sends uint8/255. Round, do not truncate a floating
        # representation of an integer pixel down to its preceding integer.
        value = value.mul(255).round().to(torch.uint8)
    return np.ascontiguousarray(value.numpy())


def build_robotwin_vae_input(views):
    """CPU FP32 [N,3,192,384], exact FACT non-antialiased three-view transform."""
    raw = {key: _rgb_thwc(views[key]) for key in VIEW_KEYS}
    lengths = {value.shape[0] for value in raw.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("camera batch lengths must agree and be nonempty")
    helper = fact_preprocess()
    # Fixed per-frame operation shapes also remove preprocessing batch effects.
    return torch.cat([helper._build_composite(
        {key: value[i:i+1] for key, value in raw.items()}, MAIN_VIEW_SIZE, list(VIEW_KEYS))
        for i in range(next(iter(lengths)))])


def encode_robotwin_observations(vae, observations):
    """Live entry point; the same encoder/rounding used when writing caches."""
    from robonana.encoding import encode_flux2_image_tokens
    if not observations:
        raise ValueError("observations must be nonempty")
    images = torch.cat([build_robotwin_vae_input(o) for o in observations])
    return encode_flux2_image_tokens(vae, images.to(next(vae.parameters()).device))


@lru_cache(maxsize=8)
def image_contract(checkpoint):
    """Fingerprint weights and runtime; old shape-only caches cannot certify parity."""
    directory = Path(checkpoint).expanduser().resolve() / "vae"
    files = sorted([*directory.glob("*.safetensors"), *directory.glob("*.bin")])
    if not files or not (directory / "config.json").is_file():
        raise FileNotFoundError(f"Missing VAE weights/config under {directory}")
    digest = hashlib.sha256()
    for path in [directory / "config.json", *files]:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return {"version": IMAGE_PIPELINE_VERSION, "vae_sha256": digest.hexdigest(),
            "fact_helper_sha256": hashlib.sha256(Path(fact_preprocess().__file__).read_bytes()).hexdigest(),
            "pixel_transform": "FACT bilinear align_corners=False antialias=False; per-view [-1,1]",
            "main_wh": [256,192], "canvas_wh": [384,192], "vae_batch": 1,
            "vae_dtype": "float32", "tf32": False, "cudnn_deterministic": True,
            "storage_dtype": "bfloat16", "model_input": "bfloat16_roundtrip_to_float32",
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("diffusers", "torchvision", "Pillow", "av")}}


def write_image_contract(task_dir, checkpoint):
    directory = Path(task_dir) / "flux_cache/latents_v2"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "_contract.json"
    expected = image_contract(str(Path(checkpoint).resolve()))
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise RuntimeError(f"Image contract changed at {path}; archive/rebuild explicitly, never relabel old caches")
        return
    if any(directory.glob("episode_*.pt")):
        raise RuntimeError(f"Uncertified image caches exist under {directory}; refusing to relabel")
    # Atomic publication with no overwrite, also safe for torchrun workers.
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".contract.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != expected:
                raise RuntimeError(f"Concurrent incompatible image contract at {path}")
    finally:
        Path(temporary).unlink(missing_ok=True)


def require_image_contract(task_dir, expected=None):
    path = Path(task_dir) / "flux_cache/latents_v2/_contract.json"
    if not path.is_file():
        raise RuntimeError(f"Image cache needs unified-pipeline rebuild: {path}. Legacy latents are not accepted.")
    actual = json.loads(path.read_text())
    if actual.get("version") != IMAGE_PIPELINE_VERSION or (expected is not None and actual != expected):
        raise RuntimeError(f"Image pipeline/weights/runtime mismatch: {path}; rebuild with the configured VAE")
    return actual


def save_image_cache(tokens, path):
    """Publish data and completion proof; never treat a partial .pt as ready."""
    path = Path(path)
    contract = require_image_contract(path.parents[2])
    value = tokens.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if value.ndim != 3 or tuple(value.shape[1:]) != (288,128) or not bool(torch.isfinite(value).all()):
        raise ValueError("Invalid image cache shape/nonfinite values")
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)
    metadata = {"contract": contract, "shape": list(value.shape), "dtype": "bfloat16"}
    temp_meta = path.with_suffix(".json.tmp")
    temp_meta.write_text(json.dumps(metadata), encoding="utf-8")
    temp_meta.replace(path.with_suffix(".json"))


def valid_image_cache(path, length=None):
    path = Path(path)
    try:
        contract = require_image_contract(path.parents[2])
        proof = json.loads(path.with_suffix(".json").read_text())
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        return (proof.get("contract") == contract and proof.get("shape") == list(value.shape)
                and value.dtype == torch.bfloat16 and value.ndim == 3
                and tuple(value.shape[1:]) == (288,128)
                and (length is None or value.shape[0] == length))
    except (OSError, RuntimeError, ValueError, KeyError, EOFError):
        return False


def validate_training_image_contracts(dataset, checkpoint):
    """Fail before training if any original/replay pool uses incompatible caches."""
    expected = image_contract(str(Path(checkpoint).resolve()))
    checked = set()
    normalization = None
    normalization_source = None
    pending = [dataset]
    visited = set()
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if hasattr(node, "_ensure_index") and hasattr(node, "records"):
            node._ensure_index()
            # A shared image pipeline is insufficient if replay changes the
            # coordinate system for state/clean action. Compare actual values,
            # not file names (historical critic configs used two stats files).
            stats = json.loads(Path(node.stats_path).read_text())["norm_stats"]
            signature = {key: {field: stats[key][field] for field in ("mean", "std")}
                         for key in ("observation.state", "action")}
            if normalization is not None and signature != normalization:
                raise RuntimeError(f"Mixed state/action normalization: {normalization_source} vs {node.stats_path}; "
                                   "use the Stage-1 policy statistics for every pool")
            normalization, normalization_source = signature, node.stats_path
            for record in node.records:
                if record.task_dir not in checked:
                    require_image_contract(record.task_dir, expected)
                    checked.add(record.task_dir)
        elif hasattr(node, "datasets"):
            pending.extend(node.datasets)
        elif hasattr(node, "dataset"):
            pending.append(node.dataset)
        else:
            raise ValueError(f"Cannot certify image pipeline for {type(node).__name__}")
    if not checked:
        raise RuntimeError("No training image contracts found; refusing uncertified input")
