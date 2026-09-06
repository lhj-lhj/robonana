"""Canonical fixed-48 MAC configuration.

Every run starts from the converted 1000-step MAC checkpoint by default. Set
ROBONANA_MAC_PRETRAIN_CHECKPOINT/CONFIG to continue from a later MAC run.
"""

from .posttrain_config import apply_mac_posttrain_config
from .robotwin_flux2 import config as _base_config


config = apply_mac_posttrain_config(_base_config)
