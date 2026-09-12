"""Regression coverage for the maintained world-fit diagnostic, not a new sampler."""
import runpy
from pathlib import Path


def test_pretrain_and_replay_pool_shapes():
    probe = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/diagnostics/probe_mac_world_fit.py"))
    pool = {"pool_name": "original_success"}
    for specification in (pool, [pool], (pool,)):
        config = {"dataloaders": {"train": {"data_or_config": specification}}}
        assert list(probe["configured_pools"](config)) == [pool]


def test_fixed_probe_selects_early_middle_and_tail():
    probe = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/diagnostics/probe_mac_world_fit.py"))
    class Dataset:
        records = [None, None]
        episode_starts = [0, 100]
        episode_stops = [100, 180]
        def __len__(self):
            return 180
    assert probe["probe_indices"](Dataset(), 2) == [0, 49, 99, 100, 139, 179]
