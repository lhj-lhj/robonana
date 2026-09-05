"""Fixed-48 MAC posttraining initialized from the immutable 120k checkpoint."""

from .posttrain_config import apply_mac_posttrain_config
from .robotwin_flux2_4b_dino import config as _base_config


config = apply_mac_posttrain_config(_base_config)
