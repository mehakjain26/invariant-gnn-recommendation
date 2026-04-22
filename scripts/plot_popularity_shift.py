"""Scatter of train-rank vs test-rank for items, colored by rising/falling/stable.
Annotates Spearman ρ. Core figure motivating the temporal popularity-shift problem.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src.data.dataset import RecDataset


def plot_popularity_shift(dataset_name="yelp2018", allowlist="ngcf",
                          k_core=0, split_mode="temporal", split_files=None,
                          seed=2020, shift_percentile=80, ax=None):
    ds = RecDataset(dataset_name, k_core=k_core, allowlist=allowlist,
                    split_mode=split_mode, split_files=split_files, seed=seed)
    train_deg = ds.item_degrees("train")
    test_deg = ds.item_degrees("test")

    # rank: higher degree → lower rank number
    train_ranks = np.argsort(np.argsort(-train_deg))
    test_ranks = np.argsort(np.argsort(-test_deg))
    rank_shift = train_ranks.astype(float) - test_ranks.astype(float)

    thresh = np.percentile(np.abs(rank_shift), shift_percentile)
    rising = rank_shift > thresh   # low train rank → high test rank
    falling = rank_shift < -thresh
    stable = ~(rising | falling)

    # keep only items present in both splits for fair scatter
    keep = (train_deg > 0) & (test_deg > 0)

    rho, _ = spearmanr(train_deg[keep], test_deg[keep])

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
    else:
        fig = ax.figure

    for mask, label, color in [
        (stable & keep, f"stable (n={(stable & keep).sum()})", "tab:gray"),
        (falling & keep, f"falling (n={(falling & keep).sum()})", "tab:red"),
        (rising & keep, f"rising (n={(rising & keep).sum()})", "tab:green"),
    ]:
        ax.scatter(train_ranks[mask], test_ranks[mask], s=4, alpha=0.4,
                   label=label, color=color)

    n = len(train_deg)
    ax.plot([0, n], [0, n], "k--", lw=0.8, alpha=0.5, label="no shift")
    ax.set_xlabel("train rank (0 = most popular)")
    ax.set_ylabel("test rank")
    ax.set_title(f"{dataset_name} popularity shift (Spearman ρ = {rho:.3f})")
    ax.legend(markerscale=3, loc="lower right")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    fig.tight_layout()
    return fig, {"rho": rho, "n_rising": int(rising.sum()),
                 "n_falling": int(falling.sum()), "n_stable": int(stable.sum())}


if __name__ == "__main__":
    for ds in ["yelp2018", "amazon-books"]:
        fig, stats = plot_popularity_shift(ds)
        fig.savefig(f"paper/figures/popularity_shift_{ds}.pdf")
        print(f"{ds}: {stats}")
