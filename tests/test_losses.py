from types import SimpleNamespace

import torch

from robonana.training.losses import joint_flow_loss, masked_bce_with_logits, masked_mse


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


def test_joint_loss_adds_dino_only_when_target_is_present():
    zeros = torch.zeros(2, 1, 1)
    output = SimpleNamespace(
        image=zeros,
        action=zeros,
        future_state=zeros,
        reward=zeros,
        success=zeros,
        q=zeros,
        dino=torch.ones(2, 3, 4),
    )
    losses = joint_flow_loss(
        output,
        image_target=zeros,
        action_target=zeros,
        future_state_target=zeros,
        reward_target=zeros,
        success_target=zeros,
        q_target=zeros,
        dino_target=torch.zeros(2, 3, 4),
    )
    assert losses["dino_loss"].item() == 1.0
    assert set(losses) == {
        "image_loss",
        "action_loss",
        "future_state_loss",
        "reward_loss",
        "success_loss",
        "q_loss",
        "dino_loss",
    }


def test_joint_loss_masks_zero_length_td_q_samples():
    zeros = torch.zeros(2, 1, 1)
    output = SimpleNamespace(
        image=zeros,
        action=zeros,
        future_state=zeros,
        reward=zeros,
        success=zeros,
        q=torch.tensor([[[2.0]], [[100.0]]]),
        dino=None,
    )
    losses = joint_flow_loss(
        output,
        image_target=zeros,
        action_target=zeros,
        future_state_target=zeros,
        reward_target=zeros,
        success_target=zeros,
        q_target=zeros,
        q_loss_mask=torch.tensor([1.0, 0.0]),
    )
    assert losses["q_loss"].item() == 4.0
