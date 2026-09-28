"""Chunk configuration controls labels, initialization, cached inference and artifacts."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch
from flux2.model import Flux2
from safetensors.torch import save_file

from test_mac_prefix_cache import model_and_inputs, sampling_inputs
from test_world_rope_prefix import world_inputs
from test_mac_dataset_contract import episode_dataset
from test_explicit_config import options
from robonana.configs.training import build_training_config
from robonana.inference_contract import sampling_contract
from robonana.models.mac_flux2_fact import MacFlux2FACTModel
from robonana.models.pretrained import load_flux2_backbone_checkpoint, load_flux2_fact_trained_checkpoint
from robonana.sampling import sample_flux2_action, sample_mac_world


@pytest.mark.parametrize('horizon', [16, 48])
def test_one_option_controls_every_training_consumer(tmp_path, horizon):
    config = build_training_config(options(tmp_path, chunk_horizon=horizon))
    data = config['dataloaders']['train']['data_or_config']
    model = config['models']
    post = config['train']['posttrain']
    assert data['action_chunk'] == data['fixed_horizon'] == data['max_horizon'] == horizon
    assert model['chunk_horizon'] == model['max_horizon'] == model['reward_dim'] == horizon
    assert model['success_dim'] == 1
    contract = sampling_contract(post)
    assert contract['action_chunk'] == contract['horizon'] == horizon
    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match='chunk_horizon'):
            options(tmp_path, chunk_horizon=invalid)
    post['environment_policy']['execute_actions_per_plan'] = horizon + 1
    with pytest.raises(ValueError, match='horizon'):
        sampling_contract(post)


@pytest.mark.parametrize('architecture,mode', [('mac_mot_v2','fixed48'), ('mac_mot_v2','rope_prefix'), ('mac_mot_v3','fixed48')])
@pytest.mark.parametrize('horizon', [16, 48])
def test_horizon_initialization_checkpoint_and_cached_forward(tmp_path, architecture, mode, horizon):
    base, condition = model_and_inputs()
    # Use the exact tiny official FLUX architecture already covered by the cache tests.
    from flux2.model import Flux2Params
    params = Flux2Params(in_channels=8, context_in_dim=16, hidden_size=32, num_heads=4,
        depth=2, depth_single_blocks=2, axes_dim=[2,2,2,2], mlp_ratio=2., use_guidance_embed=False)
    path = tmp_path / 'flux.safetensors'
    save_file(Flux2(params).state_dict(), str(path))
    model, _ = load_flux2_backbone_checkpoint(path, params=params, action_dim=6, state_dim=6,
        expert_hidden_dim=16, architecture_version=architecture, chunk_horizon=horizon,
        world_conditioning=mode, dtype=torch.float32)
    model.eval()
    inputs = world_inputs(condition, torch.full((2,), horizon))
    inputs['chunk_horizon'] = torch.full((2,), horizon)
    for key in ('noisy_pred_action','gt_action_cond'):
        inputs[key] = inputs[key][:,:horizon]
    out = model(**inputs)
    assert out.action.shape == (2,horizon,6) and out.reward.shape == (2,horizon)
    assert out.success.shape == (2,1)
    out.action.square().mean().backward()
    assert model.state_in.weight.grad.abs().sum() > 0
    cache = model.prefill_condition_cache(**condition)
    with torch.no_grad():
        cached = model.predict_action_cached(cache, inputs['noisy_pred_action'],
            batch_indices=torch.arange(2), timestep=inputs['action_timestep'])
        torch.testing.assert_close(cached, out.action, atol=3e-6, rtol=3e-5)
        args = dict(model=model, **sampling_inputs(condition), clean_action=inputs['gt_action_cond'],
            future_noise=inputs['noisy_future_latents'], future_state_noise=inputs['noisy_future_state'],
            schedule=torch.linspace(1,0,3), grid_height=1, grid_width=2)
        full = sample_mac_world(**args, use_cache=False)
        cached = sample_mac_world(**args)
        for name in ('future','future_state','reward_logits','success_logit'):
            torch.testing.assert_close(getattr(full,name),getattr(cached,name),atol=3e-6,rtol=3e-5)
        sampled = sample_flux2_action(model=model, **sampling_inputs(condition),
            action_noise=inputs['noisy_pred_action'], chunk_horizon=horizon,
            schedule=torch.linspace(1,0,3), grid_height=1, grid_width=2)
        assert sampled.shape == (2,horizon,6)
    weights = tmp_path / 'model.bin'
    torch.save(model.state_dict(), weights)
    metadata = dict(params=asdict(params), action_dim=6, state_dim=6, reward_dim=horizon,
        success_dim=1,q_dim=1,value_dim=1,max_horizon=horizon,chunk_horizon=horizon,
        architecture_version=architecture,expert_hidden_dim=16,world_conditioning=mode,
        reward_head_type='binary_chunk',include_critics=False)
    config_path = tmp_path / 'model_config.json'
    config_path.write_text(json.dumps(metadata))
    loaded,_ = load_flux2_fact_trained_checkpoint(weights,device='cpu',dtype=torch.float32)
    assert loaded.chunk_horizon == loaded.reward_dim == horizon
    torch.testing.assert_close(loaded(**inputs).action, out.action, atol=0,rtol=0)
    metadata['reward_dim'] = horizon + 1
    config_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='reward_dim'):
        load_flux2_fact_trained_checkpoint(weights,device='cpu')


@pytest.mark.parametrize('source',['hdf5','lerobot'])
@pytest.mark.parametrize('success',[True,False])
def test_16_step_dataset_endpoint_and_absorbing_labels(tmp_path,monkeypatch,source,success):
    ds, states, _, mean, std = episode_dataset(tmp_path,monkeypatch,source=source,
        success=success,length=40,chunk_horizon=16)
    try:
        row=ds._get_data(0)
        assert row['action'].shape == (16,14)
        assert row['reward_chunk'].shape == (16,)
        torch.testing.assert_close(row['future_state'],(torch.from_numpy(states[16])-mean)/std)
        assert row['future_latents'].eq(16).all()
        assert row['success'].item() == 0
        if success:
            terminal=ds._get_data(39)
            assert terminal['success'].item() == 1
            assert terminal['action_valid_mask'].all() and terminal['reward_chunk_mask'].all()
            assert terminal['reward_chunk'].eq(1).all()
            from robonana.data.robotwin_hdf5 import ALOHA_DELTA_MASK
            delta=terminal['action']*std+mean
            torch.testing.assert_close(delta[:,ALOHA_DELTA_MASK],torch.zeros(16,12),atol=1e-6,rtol=0)
        else:
            assert len(ds) == 24
    finally:
        ds.close()
