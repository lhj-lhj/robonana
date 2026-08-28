"""Iterative posttraining for the 800M+DINO RoboNana model."""

from .posttrain_config import apply_iterative_posttrain_config
from .robotwin_flux2_800m_dino import config as _base_config


config = apply_iterative_posttrain_config(_base_config)
