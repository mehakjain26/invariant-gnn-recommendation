"""Visualize popularity-based environment partition. Item-degree histogram with
vertical lines at env boundaries. Makes the 'pseudo-environments from popularity'
methodology concrete for the paper's methods section.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import RecDataset
from src.data.env_partition import partition_environments


def plot_env_partition(dataset_name="yelp2018", allowlist="ngcf", k_core=0,
                       n_envs=3, ax=None):
    ds = RecDataset(dataset_name, k_core=k_core, allowlist=allowlist, split_mode="temporal")
    train_deg = ds.item_degrees("train")
    env_ids = partition_environments(train_deg, n_envs=n_envs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    deg_positive = train_deg[train_deg > 0]
    bins = np.logspace(0, np.log10(deg_positive.max() + 1), 50)
    colors = ["tab:green", "tab:gray", "tab:red"]
    labels = [f"env {i}" for i in range(n_envs)]

    # Per-env histogram (stacked)
    for e in range(n_envs):
        mask = (env_ids == e) & (train_deg > 0)
        sizes = train_deg[mask]
        ax.hist(sizes, bins=bins, alpha=0.7, color=colors[e % len(colors)],
                label=f"{labels[e]} (n={mask.sum()}, mean deg={sizes.mean():.1f})",
                edgecolor="white", linewidth=0.3)

    # Env boundary lines = min of each env (in sorted order)
    for e in range(1, n_envs):
        mask = env_ids == e
        if mask.any():
            boundary = train_deg[mask].min()
            ax.axvline(boundary, color="black", ls="--", lw=1, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("item degree in train (log)")
    ax.set_ylabel("# items (log)")
    ax.set_title(f"{dataset_name}: env partition (E={n_envs}) by training popularity")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    for ds in ["yelp2018", "amazon-books"]:
        fig = plot_env_partition(ds)
        fig.savefig(f"paper/figures/env_partition_{ds}.pdf")
        fig.savefig(f"paper/figures/env_partition_{ds}.png", dpi=200)
        print(f"saved paper/figures/env_partition_{ds}.pdf/.png")
