"""Evaluation metrics for synthetic tasks."""

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between prediction and target.

    Args:
        pred: Predicted stream state [B, n, d]
        target: Target stream state [B, n, d]

    Returns:
        Scalar MSE loss
    """
    return F.mse_loss(pred, target)
