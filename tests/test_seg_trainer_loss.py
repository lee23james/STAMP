import torch

from train.seg_trainer import WeightedDiceBCELoss


def test_weighted_dice_bce_loss_handles_non_finite_logits():
    loss_fn = WeightedDiceBCELoss(alpha=0.3, beta=0.7)
    logits = torch.tensor([[float("nan")], [float("inf")], [float("-inf")], [0.0]])
    targets = torch.tensor([[1.0], [1.0], [0.0], [0.0]])

    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
