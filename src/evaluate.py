"""Evaluation metrics: Recall@K, NDCG@K, and subgroup analysis."""

import numpy as np
import torch
from typing import Dict, Tuple
from scipy.stats import spearmanr


def recall_at_k(ranked_items: np.ndarray, ground_truth: set, k: int) -> float:
    """Recall@K for a single user (matches LightGCN paper: hits / |gt|)."""
    if len(ground_truth) == 0:
        return 0.0
    hits = len(set(ranked_items[:k]) & ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(ranked_items: np.ndarray, ground_truth: set, k: int) -> float:
    """NDCG@K for a single user."""
    if len(ground_truth) == 0:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


@torch.no_grad()
def evaluate(
    model,
    norm_adj: torch.sparse.FloatTensor,
    train_user_items: Dict[int, set],
    test_user_items: Dict[int, set],
    k: int = 20,
    batch_size: int = 256,
) -> Dict[str, float]:
    """Evaluate model on test set.

    Args:
        model: LightGCN model.
        norm_adj: Normalized adjacency (on device).
        train_user_items: {user: set(items)} from training (to exclude).
        test_user_items: {user: set(items)} ground truth.
        k: Cutoff for metrics.
        batch_size: Users per batch for scoring.

    Returns:
        Dict with recall@k and ndcg@k.
    """
    model.eval()
    user_emb, item_emb = model(norm_adj)

    users = sorted(test_user_items.keys())
    recalls, ndcgs = [], []

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        u_emb = user_emb[batch_users]  # (B, d)
        scores = u_emb @ item_emb.t()  # (B, n_items)

        # Mask out training items
        for idx, u in enumerate(batch_users):
            train_items = train_user_items.get(u, set())
            if train_items:
                scores[idx, list(train_items)] = -float("inf")

        # Top-K
        _, topk_indices = scores.topk(k, dim=1)
        topk_indices = topk_indices.cpu().numpy()

        for idx, u in enumerate(batch_users):
            gt = test_user_items[u]
            recalls.append(recall_at_k(topk_indices[idx], gt, k))
            ndcgs.append(ndcg_at_k(topk_indices[idx], gt, k))

    return {
        f"recall@{k}": np.mean(recalls),
        f"ndcg@{k}": np.mean(ndcgs),
    }


@torch.no_grad()
def evaluate_subgroups(
    model,
    norm_adj: torch.sparse.FloatTensor,
    train_user_items: Dict[int, set],
    test_user_items: Dict[int, set],
    train_item_degrees: np.ndarray,
    test_item_degrees: np.ndarray,
    k: int = 20,
    batch_size: int = 256,
) -> Dict[str, Dict[str, float]]:
    """Evaluate with subgroup breakdown by popularity trajectory.

    Items are categorized as:
    - Rising: degree rank increased substantially (train → test)
    - Falling: degree rank decreased substantially
    - Stable: roughly same rank

    Returns:
        Dict mapping subgroup name → {recall@k, ndcg@k}.
    """
    model.eval()
    user_emb, item_emb = model(norm_adj)
    n_items = len(train_item_degrees)

    # Compute rank shifts
    train_ranks = np.argsort(np.argsort(-train_item_degrees))  # higher degree = lower rank
    test_ranks = np.argsort(np.argsort(-test_item_degrees))
    rank_shift = train_ranks.astype(float) - test_ranks.astype(float)  # positive = item rose

    # Classify items by shift magnitude (top/bottom 20% of shift)
    shift_thresh = np.percentile(np.abs(rank_shift), 80)
    rising_items = set(np.where(rank_shift > shift_thresh)[0])
    falling_items = set(np.where(rank_shift < -shift_thresh)[0])
    stable_items = set(range(n_items)) - rising_items - falling_items

    subgroups = {"rising": rising_items, "falling": falling_items, "stable": stable_items}

    # Spearman correlation (only on items present in both to prevent zero-tie distortion)
    keep = (train_item_degrees > 0) & (test_item_degrees > 0)
    if keep.sum() > 1:
        rho, _ = spearmanr(train_item_degrees[keep], test_item_degrees[keep])
    else:
        rho = 0.0

    results = {"spearman_rho": rho}

    users = sorted(test_user_items.keys())

    for group_name, group_items in subgroups.items():
        recalls, ndcgs = [], []

        for start in range(0, len(users), batch_size):
            batch_users = users[start : start + batch_size]
            u_emb = user_emb[batch_users]
            scores = u_emb @ item_emb.t()

            for idx, u in enumerate(batch_users):
                train_items = train_user_items.get(u, set())
                if train_items:
                    scores[idx, list(train_items)] = -float("inf")

            _, topk_indices = scores.topk(k, dim=1)
            topk_indices = topk_indices.cpu().numpy()

            for idx, u in enumerate(batch_users):
                # Only count hits in this subgroup
                gt = test_user_items[u] & group_items
                if len(gt) == 0:
                    continue
                recalls.append(recall_at_k(topk_indices[idx], gt, k))
                ndcgs.append(ndcg_at_k(topk_indices[idx], gt, k))

        results[group_name] = {
            f"recall@{k}": np.mean(recalls) if recalls else 0.0,
            f"ndcg@{k}": np.mean(ndcgs) if ndcgs else 0.0,
            "n_items": len(group_items),
        }

    return results
