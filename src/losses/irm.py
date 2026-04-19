"""IRM (Invariant Risk Minimization) penalty for BPR."""

import torch
import torch.nn.functional as F
from typing import List


def irm_penalty(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    env_masks: List[torch.Tensor],
) -> torch.Tensor:
    """Compute IRM penalty: sum_e ||grad_{w=1} R^e||^2.

    We introduce a dummy scalar w=1.0 that scales the score difference.
    The IRM penalty is the squared gradient of each per-environment BPR
    risk with respect to w, evaluated at w=1. This enforces that w=1 is
    simultaneously optimal for all environments.

    Args:
        pos_scores: Positive item scores, shape (batch,).
        neg_scores: Negative item scores, shape (batch,).
        env_masks: List of boolean masks, one per environment.

    Returns:
        Scalar IRM penalty.
    """
    # Dummy scalar multiplier
    w = torch.tensor(1.0, device=pos_scores.device, requires_grad=True)

    penalty = torch.tensor(0.0, device=pos_scores.device)
    for mask in env_masks:
        if mask.sum() == 0:
            continue
        diff = w * (pos_scores[mask] - neg_scores[mask])
        risk = -F.logsigmoid(diff).mean()
        grad = torch.autograd.grad(risk, w, create_graph=True)[0]
        penalty = penalty + grad.pow(2)

    return penalty
