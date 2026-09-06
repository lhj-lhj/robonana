import torch

from robonana.data.robotwin_hdf5 import mac_binary_chunk_targets


def test_fixed_mac_reward_chunk_uses_absorbing_success_suffix():
    target, mask = mac_binary_chunk_targets(
        delta_steps=3, success_terminal=True, chunk_horizon=48
    )
    assert target.shape == (48,)
    assert mask.shape == (48,)
    torch.testing.assert_close(target[:3], torch.zeros(3))
    torch.testing.assert_close(target[3:], torch.ones(45))
    torch.testing.assert_close(mask, torch.ones(48))


def test_failed_tail_is_unknown_and_not_padded():
    target, mask = mac_binary_chunk_targets(
        delta_steps=48, success_terminal=False, chunk_horizon=48
    )
    torch.testing.assert_close(target, torch.zeros(48))
    torch.testing.assert_close(mask, torch.ones(48))
