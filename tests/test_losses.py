import torch

from robonana.training.losses import masked_mse


def test_failure_mask_removes_action_sample():
    prediction = torch.tensor([[[1.0]], [[100.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = masked_mse(prediction, target, torch.tensor([1.0, 0.0]))
    loss.backward()
    assert loss.item() == 1.0
    assert prediction.grad[0].abs().sum() > 0
    assert prediction.grad[1].abs().sum() == 0

