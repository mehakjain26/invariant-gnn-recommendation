"""Partition items into popularity-based environments."""

import numpy as np
import torch
from typing import List, Tuple


def partition_environments(
    item_degrees: np.ndarray,
    n_envs: int = 3,
    percentiles: List[float] = None,
) -> np.ndarray:
    """Assign each item to a popularity environment based on degree percentile.

    Args:
        item_degrees: Array of shape (n_items,) with degree counts.
        n_envs: Number of environments (2, 3, or 5).
        percentiles: Custom percentile boundaries. If None, uses defaults:
            - 2 envs: [50] → bottom-50%, top-50%
            - 3 envs: [20, 80] → bottom-20%, mid-60%, top-20%
            - 5 envs: [10, 30, 60, 90] → 5 buckets

    Returns:
        env_ids: Array of shape (n_items,) with environment index per item.
    """
    if percentiles is None:
        defaults = {
            2: [50],
            3: [20, 80],
            5: [10, 30, 60, 90],
        }
        if n_envs not in defaults:
            raise ValueError(f"n_envs must be 2, 3, or 5 (got {n_envs})")
        percentiles = defaults[n_envs]

    thresholds = np.percentile(item_degrees, percentiles)
    env_ids = np.digitize(item_degrees, thresholds)
    return env_ids


def assign_triplet_envs(
    pos_items: torch.Tensor,
    item_env_ids: np.ndarray,
) -> torch.Tensor:
    """Assign each BPR triplet to the environment of its positive item.

    Args:
        pos_items: Tensor of positive item indices, shape (batch_size,).
        item_env_ids: Array mapping item index → environment id.

    Returns:
        Tensor of environment ids, shape (batch_size,).
    """
    return torch.LongTensor(item_env_ids[pos_items.cpu().numpy()])


def get_env_masks(
    env_ids: torch.Tensor,
    n_envs: int,
) -> List[torch.Tensor]:
    """Return a list of boolean masks, one per environment.

    Args:
        env_ids: Tensor of environment ids, shape (batch_size,).
        n_envs: Number of environments.

    Returns:
        List of boolean tensors, each of shape (batch_size,).
    """
    return [env_ids == e for e in range(n_envs)]


def print_env_stats(item_degrees: np.ndarray, env_ids: np.ndarray) -> None:
    """Print summary statistics for each environment."""
    for e in range(env_ids.max() + 1):
        mask = env_ids == e
        n_items = mask.sum()
        mean_deg = item_degrees[mask].mean() if n_items > 0 else 0
        print(f"  Env {e}: {n_items} items, mean degree = {mean_deg:.1f}")
