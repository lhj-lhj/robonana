"""Iterative posttraining for the pretrained FLUX.2 Klein 4B+DINO model."""

from .posttrain_config import apply_iterative_posttrain_config
from .robotwin_flux2_4b_dino import config as _base_config


config = apply_iterative_posttrain_config(_base_config)
