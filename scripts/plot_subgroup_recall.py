"""Bar/line chart of Recall@20 per subgroup (rising/falling/stable) × method × λ.
Core experimental figure. Loads from results/*/results.json.
"""

import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir="results", dataset="yelp2018", split_tag=None):
    """Load all results.json for a dataset's temporal sweep.
    Structure: {results_dir}/layer_3/{dataset_key}/{baseline_temporal,irm_lamX,vrex_lamX}/results.json
    """
    # dataset key in folder tree
    ds_key = "yelp" if dataset.startswith("yelp") else "amazon"
    pattern = f"{results_dir}/layer_3/{ds_key}/*/results.json"
    out = []
    for p in sorted(glob.glob(pattern)):
        name = Path(p).parent.name
        if name == "baseline_temporal":
            out.append({"method": "baseline", "lam": None, "r": json.load(open(p))})
            continue
        m = re.match(r"(irm|vrex)_lam([\d.]+)", name)
        if not m:
            continue  # skip random-split baselines
        method, lam = m.group(1), float(m.group(2))
        out.append({"method": method, "lam": lam, "r": json.load(open(p))})
    return out


def plot_subgroup_recall(results, k=20, title=""):
    """Grid: rows=methods, cols=subgroups (rising/falling/stable). Each cell has its own y-axis
    so rising (≈1e-4) is visible alongside stable (≈4e-2). Baseline shown as dashed line."""
    methods = sorted({x["method"] for x in results if x["method"] != "baseline"})
    baseline = next((x for x in results if x["method"] == "baseline"), None)
    subs = ["rising", "falling", "stable"]
    colors = {"rising": "tab:green", "falling": "tab:red", "stable": "tab:gray"}

    n_rows, n_cols = len(methods), len(subs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for r, method in enumerate(methods):
        runs = sorted([x for x in results if x["method"] == method], key=lambda x: x["lam"])
        lams = [x["lam"] for x in runs]
        x_pos = np.arange(len(lams))

        for c, sub in enumerate(subs):
            ax = axes[r, c]
            vals = [x["r"]["subgroups"][sub][f"recall@{k}"] for x in runs]
            ax.bar(x_pos, vals, color=colors[sub], alpha=0.85, edgecolor="black", linewidth=0.5)

            if baseline:
                bval = baseline["r"]["subgroups"][sub][f"recall@{k}"]
                # pick decimal precision based on magnitude so label isn't misleadingly rounded
                if bval < 1e-3:
                    blabel = f"baseline ({bval:.2e})"
                else:
                    blabel = f"baseline ({bval:.4f})"
                ax.axhline(bval, ls="--", color="black", alpha=0.7, lw=1.2, label=blabel)
                ax.legend(fontsize=8, loc="best")

            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"λ={l}" for l in lams])
            ax.set_title(f"{method.upper()} — {sub}")
            if c == 0:
                ax.set_ylabel(f"Recall@{k}")
            ax.grid(axis="y", alpha=0.3)
            # give rising a tiny headroom so bars aren't squashed
            if sub == "rising":
                ymax = max(max(vals), baseline["r"]["subgroups"][sub][f"recall@{k}"] if baseline else 0)
                ax.set_ylim(0, ymax * 1.3 if ymax > 0 else 1e-3)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for ds in ["yelp2018", "amazon-books"]:
        results = load_results(dataset=ds)
        if not results:
            print(f"no results for {ds}")
            continue
        fig = plot_subgroup_recall(results, title=ds)
        fig.savefig(f"paper/figures/subgroup_recall_{ds}.pdf")
        fig.savefig(f"paper/figures/subgroup_recall_{ds}.png", dpi=200)
        print(f"saved paper/figures/subgroup_recall_{ds}.pdf/.png ({len(results)} runs)")
