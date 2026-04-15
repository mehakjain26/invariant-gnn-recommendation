"""Bayesian Personalized Ranking (BPR) loss."""

import torch
import torch.nn.functional as F


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Compute BPR loss: -log(sigmoid(pos - neg)).

    Args:
        pos_scores: Scores for positive items, shape (batch,).
        neg_scores: Scores for negative items, shape (batch,).
        mask: Optional boolean mask to select a subset (for per-env loss).

    Returns:
        Scalar BPR loss.
    """
    diff = pos_scores - neg_scores
    if mask is not None:
        diff = diff[mask]
    if len(diff) == 0:
        return torch.tensor(0.0, device=pos_scores.device, requires_grad=True)
    return -F.logsigmoid(diff).mean()
