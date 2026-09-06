"""Bounded hanging-mug pilot; retain every checkpoint for offline probes.

All model/loss/data semantics come from the maintained MAC config. This only
sets experiment budget and checkpoint retention, not an alternative trainer.
"""

import copy

from .robotwin_flux2_4b_mac import config as _mac_config


def apply_pilot_config(base):
    config = copy.deepcopy(base)
    phase = config["train"]["posttrain"]["phase"]
    steps = 5000 if phase == "world_policy" else 500
    config["train"].update(
        max_steps=steps, checkpoint_interval=1000 if phase == "world_policy" else 250,
        early_checkpoint_steps=(500,) if phase == "world_policy" else (), seed=20260906,
        checkpoint_total_limit=8, log_interval=10, pixel_eval_interval=0,
    )
    config["schedulers"].update(warmup_steps=250 if phase == "world_policy" else 100,
                                 decay_steps=steps)
    if phase == "critic":
        # The previous 1-2 step smoke had no warmup. Use a conservative pilot
        # learning rate; this is a hypothesis to test, not a convergence claim.
        config["optimizers"].update(lr=1e-5, robot_lr=1e-5)
    return config


config = apply_pilot_config(_mac_config)
