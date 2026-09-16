"""Two-arm world ablation: real dependencies, supervision and saved semantics."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test_mac_prefix_cache import model_and_inputs, sampling_inputs
from robonana.models.attention_mask import MacSegmentMap, build_mac_attention_bias
from robonana.models.position_ids import image_position_ids
from robonana.sampling import sample_mac_world


def world_inputs(inputs, h):
    return dict(**inputs, noisy_future_latents=torch.randn(2, 2, 8),
                future_ids=image_position_ids(2, grid_height=1, grid_width=2, time_coord=h, device="cpu"),
                noisy_future_state=torch.randn(2, 1, 6), noisy_pred_action=torch.randn(2, 48, 6),
                gt_action_cond=torch.randn(2, 48, 6), chunk_horizon=torch.full((2,), 48),
                world_horizon=h, noisy_reward=torch.empty(2, 0, 1), noisy_q=torch.empty(2, 0, 1),
                action_timestep=torch.full((2,), .4), wm_timestep=torch.full((2,), .7))


def test_mask_has_only_two_modes_and_blocks_all_world_suffix_paths():
    s = MacSegmentMap.from_lengths(language=3, state=1, ref_image=2, pred_action=48,
        clean_action=48, reward=1, success=1, future_state=1, future_image=2)
    opts = dict(batch_size=2, dtype=torch.float32, device="cpu")
    base = build_mac_attention_bias(s, **opts)
    h = torch.tensor([1, 48])
    new = build_mac_attention_bias(s, **opts, world_conditioning="rope_prefix", world_horizon=h)
    assert torch.equal(base, build_mac_attention_bias(s, **opts, world_conditioning="fixed48"))
    assert torch.isfinite(base[:, :, s.clean_action, s.clean_action]).all()
    assert torch.isfinite(new[:, :, s.pred_action, s.pred_action]).all()
    for b, horizon in enumerate(h.tolist()):
        clean = torch.isfinite(new[b, 0, s.clean_action, s.clean_action])
        assert torch.equal(clean, torch.ones(48, 48, dtype=torch.bool).tril())
        world = new[b, 0, s.reward.start:, s.clean_action]
        assert torch.isfinite(world[:, :horizon]).all()
        assert torch.isneginf(world[:, horizon:]).all()
        assert torch.isneginf(new[b, 0, s.clean_condition, s.clean_action]).all()
        assert torch.isneginf(new[b, 0, s.reward.start:, s.pred_action]).all()
    for invalid in (torch.tensor([0, 48]), torch.tensor([1, 49]), h.float(), h[:, None]):
        with pytest.raises(ValueError):
            build_mac_attention_bias(s, **opts, world_conditioning="rope_prefix", world_horizon=invalid)


def test_full_multilayer_world_has_no_suffix_action_value_or_gradient_leak():
    model, inputs = model_and_inputs()
    model.world_conditioning = "rope_prefix"
    h = torch.tensor([1, 17])
    kwargs = world_inputs(inputs, h)
    kwargs["gt_action_cond"].requires_grad_()
    kwargs["noisy_pred_action"].requires_grad_()
    output = model(**kwargs)
    changed = kwargs["gt_action_cond"].detach().clone()
    for b, horizon in enumerate(h.tolist()):
        changed[b, horizon:] += 100
    other = model(**{**kwargs, "gt_action_cond": changed})
    for field in ("image", "future_state", "reward", "success", "action"):
        torch.testing.assert_close(getattr(other, field), getattr(output, field), atol=0, rtol=0)
    loss = sum(getattr(output, field).square().sum() for field in ("image", "future_state", "reward", "success"))
    loss.backward()
    grad = kwargs["gt_action_cond"].grad
    for b, horizon in enumerate(h.tolist()):
        assert grad[b, :horizon].abs().sum() > 0
        assert torch.count_nonzero(grad[b, horizon:]) == 0
    assert torch.count_nonzero(kwargs["noisy_pred_action"].grad) == 0


def test_rope_coordinates_match_h_without_new_parameters():
    model, inputs = model_and_inputs()
    before = set(model.state_dict())
    model.world_conditioning = "rope_prefix"
    kwargs = world_inputs(inputs, torch.tensor([3, 48]))
    coordinates = []
    hook = model.pe_embedder.register_forward_pre_hook(lambda _module, args: coordinates.append(args[0].clone()))
    out = model(**kwargs)
    hook.remove()
    ids = coordinates[0]
    offset = inputs["context"].shape[1]
    for part in (out.segments.reward, out.segments.success, out.segments.future_state):
        assert torch.equal(ids[:, part.start-offset:part.stop-offset, 1], kwargs["world_horizon"][:, None])
    assert torch.equal(ids[:, -2:, 0], kwargs["world_horizon"][:, None].expand(-1, 2))
    assert set(model.state_dict()) == before
    assert not any("horizon_embed" in name for name in before)
    with pytest.raises(ValueError, match="RoPE time"):
        model(**{**kwargs, "world_horizon": torch.tensor([4, 48])})


@pytest.mark.parametrize("mode", ["fixed48", "rope_prefix"])
def test_world_cache_matches_full_h48_for_both_modes(mode):
    model, inputs = model_and_inputs()
    model.world_conditioning = mode
    kwargs = dict(model=model, **sampling_inputs(inputs), clean_action=torch.randn(2, 48, 6),
        future_noise=torch.randn(2, 2, 8), future_state_noise=torch.randn(2, 1, 6),
        schedule=torch.linspace(1, 0, 5), grid_height=1, grid_width=2)
    full = sample_mac_world(**kwargs, use_cache=False)
    cached = sample_mac_world(**kwargs)
    for field in ("future", "future_state", "reward_logits", "success_logit"):
        torch.testing.assert_close(getattr(cached, field), getattr(full, field), atol=3e-6, rtol=3e-5)


def make_dataset(*, success=True, length=61, mode="rope_prefix"):
    from robonana.data.robotwin_hdf5 import RoboTwinHDF5Dataset, EpisodeRecord
    ds = RoboTwinHDF5Dataset("/unused", stats_path="/unused", action_dim=6, world_conditioning=mode)
    record = EpisodeRecord("task", Path("/unused"), Path("/unused/episode0.hdf5"), 0, length,
                           success=success, time_limit_truncated=not success)
    ds._set_records([record])
    ds._stats = {"norm_stats": {key: dict(mean=[0]*6, std=[1]*6) for key in ("action", "observation.state")}}
    vector = np.arange(length, dtype=np.float32)[:, None].repeat(6, axis=1)
    valid = np.ones(length, dtype=bool)
    valid[-1] = False
    ds._episode_transition_valid = lambda _: valid
    ds._episode_state_action = lambda _: (vector, vector + 1)
    ds._latents = lambda _: torch.arange(length).float()[:, None, None].expand(-1, 2, 8)
    ds._context = lambda _: torch.zeros(3, 16)
    return ds, valid


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("h", [1, 17, 48])
def test_dataset_targets_and_bc_lengths(success, h):
    ds, _ = make_dataset(success=success)
    with patch.object(ds, "_sample_horizon", return_value=h):
        row = ds._get_data(2)
    assert row["world_horizon"].item() == h and row["chunk_horizon"].item() == 48
    assert row["future_index"].item() == 2 + h
    assert torch.all(row["future_state"] == 2 + h)
    assert torch.all(row["future_latents"] == 2 + h)
    assert row["action"].shape == (48, 6) and row["action_valid_mask"].all()
    assert row["action_loss_mask"].item() == float(success)
    assert row["reward_chunk_mask"].sum().item() == h
    assert row["reward_chunk_mask"][h:].sum() == 0
    assert row["success"].item() == 0


def test_absorbing_success_and_missing_failure_transition_after_h():
    ds, _ = make_dataset(length=11)
    with patch.object(ds, "_sample_horizon", return_value=6):
        row = ds._get_data(8)
    assert row["future_index"].item() == 10 and row["success"].item() == 1
    assert row["action_valid_mask"].sum().item() == 2
    assert row["reward_chunk_mask"].sum().item() == 6
    assert row["reward_chunk"][:2].sum() == 0
    assert torch.all(row["reward_chunk"][2:6] == 1)
    ds, valid = make_dataset(success=False)
    valid[30] = False  # h=1 must not hide an invalid full BC/action window.
    with patch.object(ds, "_sample_horizon", return_value=1), pytest.raises(RuntimeError, match="missing transitions"):
        ds._get_data(0)


def test_uniform_draw_reproducible_and_fixed48_never_consumes_rng():
    ds, _ = make_dataset()
    torch.manual_seed(17)
    got = [ds._sample_horizon() for _ in range(200)]
    torch.manual_seed(17)
    assert got == torch.randint(1, 49, (200,)).tolist()
    assert {1, 48} <= set(got)
    ds.world_conditioning = "fixed48"
    rng = torch.get_rng_state().clone()
    assert ds._sample_horizon() == 48 and torch.equal(rng, torch.get_rng_state())


def test_saved_checkpoint_restores_mode_and_legacy_defaults(tmp_path):
    from test_pretrained import tiny_model, tiny_params
    from robonana.models.pretrained import load_flux2_fact_trained_checkpoint
    expected = tiny_model()
    ckpt = tmp_path / "model.bin"
    torch.save(expected.state_dict(), ckpt)
    metadata = dict(params=asdict(tiny_params()), action_dim=6, state_dim=5, reward_dim=48,
        success_dim=1, q_dim=1, value_dim=1, reward_head_type="binary_chunk", max_horizon=48,
        architecture_version="mac_mot_v2", chunk_horizon=48, expert_hidden_dim=32)
    for mode in (None, "rope_prefix"):
        if mode:
            metadata["world_conditioning"] = mode
        (tmp_path / "model_config.json").write_text(json.dumps(metadata))
        actual, _ = load_flux2_fact_trained_checkpoint(ckpt, device="cpu", dtype=torch.float32)
        assert actual.world_conditioning == (mode or "fixed48")
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value, atol=0, rtol=0)
    (tmp_path / "inference_contract.json").write_text(json.dumps({"world_conditioning": "fixed48"}))
    with pytest.raises(ValueError, match="world_conditioning disagree"):
        load_flux2_fact_trained_checkpoint(ckpt, device="cpu", dtype=torch.float32)
