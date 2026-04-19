"""VREx (Variance Risk Extrapolation) penalty for BPR."""

import torch
import torch.nn.functional as F
from typing import List


def vrex_penalty(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    env_masks: List[torch.Tensor],
) -> torch.Tensor:
    """Compute VREx penalty: Var([R^1, ..., R^E]).

    Penalizes the variance of per-environment BPR risks, encouraging
    the model to perform equally well across all popularity environments.

    Args:
        pos_scores: Positive item scores, shape (batch,).
        neg_scores: Negative item scores, shape (batch,).
        env_masks: List of boolean masks, one per environment.

    Returns:
        Scalar VREx penalty.
    """
    risks = []
    for mask in env_masks:
        if mask.sum() == 0:
            continue
        diff = pos_scores[mask] - neg_scores[mask]
        risk = -F.logsigmoid(diff).mean()
        risks.append(risk)

    if len(risks) < 2:
        return torch.tensor(0.0, device=pos_scores.device)

    risks = torch.stack(risks)
    return risks.var()
