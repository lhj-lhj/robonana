import torch

from robonana.training.losses import (
    masked_action_mse,
    deterministic_return_loss,
    masked_bce_with_logits,
    masked_elementwise_bce_with_logits,
    masked_mse,
)


def test_absorbing_padding_and_failed_actions_have_zero_bc_gradient():
    prediction = torch.tensor([[[1.0], [100.0]], [[100.0], [100.0]]], requires_grad=True)
    loss = masked_action_mse(prediction, torch.zeros_like(prediction),
                             torch.tensor([[1, 0], [1, 1]]), torch.tensor([1, 0]))
    assert loss.item() == 1
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[2.0], [0.0]], [[0.0], [0.0]]]))


def test_failure_mask_removes_action_sample():
    prediction = torch.tensor([[[1.0]], [[100.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = masked_mse(prediction, target, torch.tensor([1.0, 0.0]))
    loss.backward()
    assert loss.item() == 1.0
    assert prediction.grad[0].abs().sum() > 0
    assert prediction.grad[1].abs().sum() == 0


def test_reward_is_binary_logit_loss():
    logits = torch.tensor([[-2.0], [2.0]])
    targets = torch.tensor([[0.0], [1.0]])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    torch.testing.assert_close(masked_bce_with_logits(logits, targets), expected)


def test_chunk_reward_bce_ignores_invalid_timeout_tail():
    logits = torch.tensor([[0.0, 0.0, 100.0]])
    targets = torch.zeros_like(logits)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :2], targets[:, :2]
    )
    torch.testing.assert_close(
        masked_elementwise_bce_with_logits(logits, targets, mask), expected
    )


def test_deterministic_return_loss_uses_fixed_scale():
    prediction = torch.tensor([[0.5], [-0.5]])
    target = torch.tensor([[500.0], [-500.0]])
    assert deterministic_return_loss(
        prediction, target, return_scale=1000.0
    ).item() == 0.0
