"""Thin FACT transform extension.

FACT remains responsible for image layout, action/state normalization, value,
failure flags, and action-loss masking.  This subclass only makes the sampled
horizon explicit for the FLUX model.
"""

import torch

from world_action_model.transformers.wa_transforms_lerobot import WATransformsLerobot


class RoboNanaTransforms(WATransformsLerobot):
    """Reuse FACT preprocessing and add ``horizon_idx`` to its output."""

    def __call__(self, data_dict):
        output = super().__call__(data_dict)
        frame = output["frame_index"].to(dtype=torch.long)
        future = output["future_state_index"].to(dtype=torch.long)
        output["horizon_idx"] = (future - frame).clamp_min(1)
        return output

