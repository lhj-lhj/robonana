"""robonana: a minimal FACT + FLUX.2 shared-backbone scaffold."""

from .models.attention_mask import SegmentMap, build_attention_bias
from .models.flux2_fact import Flux2FACTModel, Flux2FACTOutput

__all__ = [
    "Flux2FACTModel",
    "Flux2FACTOutput",
    "SegmentMap",
    "build_attention_bias",
]

