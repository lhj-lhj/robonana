"""robonana: a minimal FACT + FLUX.2 shared-backbone scaffold.

The public model symbols are loaded lazily so the lightweight RoboTwin client
can import data-recording utilities without installing the training stack.
"""

from importlib import import_module

__all__ = [
    "Flux2FACTModel",
    "Flux2FACTOutput",
    "SegmentMap",
    "build_attention_bias",
    "configure_trainable_parameters",
    "load_flux2_fact_checkpoint",
    "load_flux2_fact_trained_checkpoint",
]


def __getattr__(name: str):
    if name in {"SegmentMap", "build_attention_bias"}:
        return getattr(import_module(".models.attention_mask", __name__), name)
    if name in {"Flux2FACTModel", "Flux2FACTOutput"}:
        return getattr(import_module(".models.flux2_fact", __name__), name)
    if name in {
        "configure_trainable_parameters",
        "load_flux2_fact_checkpoint",
        "load_flux2_fact_trained_checkpoint",
    }:
        return getattr(import_module(".models.pretrained", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
