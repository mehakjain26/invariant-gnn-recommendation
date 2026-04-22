"""Histogram of interactions per month. Confirms the temporal structure is real
(not uniformly random) — justifies the temporal-split setup.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.data.dataset import RecDataset


def plot_interactions_over_time(dataset_name="yelp2018", allowlist="ngcf",
                                k_core=0, freq="1mo", ax=None):
    # Use the dataset's interaction df (post-filter, post-remap) so it matches experiments.
    ds = RecDataset(dataset_name, k_core=k_core, allowlist=allowlist, split_mode="temporal")
    # Concatenate all splits (they're disjoint temporally)
    df = pl.concat([ds.train_df, ds.val_df, ds.test_df]).sort("timestamp")

    # timestamp is unix seconds
    ts = df["timestamp"].to_numpy()
    dates = np.array([np.datetime64(int(t), "s") for t in ts])

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    # monthly bins
    start = dates.min().astype("datetime64[M]")
    end = dates.max().astype("datetime64[M]") + np.timedelta64(1, "M")
    bins = np.arange(start, end + np.timedelta64(1, "M"), np.timedelta64(1, "M"))
    ax.hist(dates, bins=bins, color="tab:blue", alpha=0.8, edgecolor="white", linewidth=0.3)

    # Mark train/val/test split boundaries
    train_end = ds.train_df["timestamp"].max()
    val_end = ds.val_df["timestamp"].max()
    for cutoff, label, color in [(train_end, "train/val", "orange"),
                                  (val_end, "val/test", "red")]:
        cutoff_date = np.datetime64(int(cutoff), "s")
        ax.axvline(cutoff_date, color=color, ls="--", lw=1.2, alpha=0.8,
                   label=f"{label} cutoff")

    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=11, rotation=0)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("year", fontsize=12)
    ax.set_ylabel("# interactions / month", fontsize=12)
    ax.set_ylabel("# interactions / month")
    ax.set_title(f"{dataset_name}: interactions per month (n={len(df):,})", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    for ds in ["yelp2018", "amazon-books"]:
        fig = plot_interactions_over_time(ds)
        fig.savefig(f"paper/figures/interactions_over_time_{ds}.pdf")
        fig.savefig(f"paper/figures/interactions_over_time_{ds}.png", dpi=200)
        print(f"saved paper/figures/interactions_over_time_{ds}.pdf/.png")
