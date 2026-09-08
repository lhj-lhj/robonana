"""One-off export guards; no legacy sampling implementation is maintained."""
import importlib.util
from pathlib import Path

import pytest
import torch

from robonana.models.flux2_fact import Flux2FACTModel
from robonana.models.mac_flux2_fact import MacFlux2FACTModel
from test_mac_prefix_cache import model_and_inputs, full_action

spec = importlib.util.spec_from_file_location('convert_actor', Path(__file__).resolve().parents[1] /
    'scripts/data/convert_120k_action_checkpoint.py')
conversion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversion)


def test_actor_migration_preserves_masked_forward_and_segment_rows():
    mac, inputs = model_and_inputs()
    # Same shared FACT wrapper as historical actor, including asymmetric mask.
    from flux2.model import Flux2Params
    params = Flux2Params(in_channels=8, context_in_dim=16, hidden_size=32, num_heads=4,
        depth=2, depth_single_blocks=2, axes_dim=[2, 2, 2, 2], mlp_ratio=2., use_guidance_embed=False)
    old = Flux2FACTModel(params, action_dim=6, state_dim=6, max_horizon=48,
                        pred_action_bidirectional=True).eval()
    source = {k: v for k, v in old.state_dict().items() if not k.startswith(
        ('reward_token.', 'success_token.', 'reward_out.', 'success_out.', 'q_in.', 'q_out.', 'q_segment_embed.'))}
    source['value_in.weight'] = torch.zeros(32, 1)
    source['value_out.weight'] = torch.zeros(1, 32)
    conversion.migrate_actor(source, mac)
    torch.testing.assert_close(mac.actor_world_segment_embed.weight[:4], old.segment_embed.weight[:4], rtol=0, atol=0)
    action = torch.randn(2, 48, 6)
    common = dict(**inputs, noisy_future_latents=torch.randn(2, 2, 8),
        future_ids=inputs['current_ids'], noisy_pred_action=action,
        gt_action_cond=torch.randn_like(action), chunk_horizon=torch.full((2,), 48),
        noisy_future_state=torch.randn(2, 1, 6), noisy_reward=torch.randn(2, 1, 1),
        noisy_q=torch.randn(2, 1, 1), wm_timestep=torch.ones(2))
    with torch.no_grad():
        for sigma in (0., .5, 1.):
            reference = old(**common, action_timestep=torch.full((2,), sigma)).action
            actual = full_action(mac, inputs, action, torch.tensor(sigma))
            torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    broken = dict(source)
    del broken['action_in.weight']
    with pytest.raises(ValueError, match='Missing actor'):
        conversion.migrate_actor(broken, mac)
    with pytest.raises(ValueError, match='Unrecognized'):
        conversion.migrate_actor(dict(source, unknown=torch.zeros(1)), mac)
