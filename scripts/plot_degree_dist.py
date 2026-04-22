"""Plot item degree distribution, train vs test (log-log). Shows long-tail + shift."""

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import RecDataset


def plot_degree_distribution(dataset_name="yelp2018", allowlist="ngcf",
                             k_core=0, split_mode="temporal", split_files=None,
                             seed=2020, ax=None):
    ds = RecDataset(dataset_name, k_core=k_core, allowlist=allowlist,
                    split_mode=split_mode, split_files=split_files, seed=seed)
    train_deg = ds.item_degrees("train")
    test_deg = ds.item_degrees("test")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    bins = np.logspace(0, np.log10(max(train_deg.max(), test_deg.max()) + 1), 40)
    ax.hist(train_deg[train_deg > 0], bins=bins, alpha=0.55, label=f"train (n={(train_deg > 0).sum()})",
            color="tab:blue")
    ax.hist(test_deg[test_deg > 0], bins=bins, alpha=0.55, label=f"test (n={(test_deg > 0).sum()})",
            color="tab:orange")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("item degree (log)")
    ax.set_ylabel("# items (log)")
    ax.set_title(f"{dataset_name} item degree distribution ({split_mode})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for ds in ["yelp2018", "amazon-books"]:
        fig = plot_degree_distribution(ds)
        fig.savefig(f"paper/figures/degree_dist_{ds}.pdf")
        print(f"saved paper/figures/degree_dist_{ds}.pdf")
