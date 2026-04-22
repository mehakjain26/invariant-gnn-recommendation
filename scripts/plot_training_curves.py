"""Plot val R@20 / N@20 training curves per method × λ. Parses training .log files.

Usage:
    # On porsche where logs live:
    python -m scripts.plot_training_curves --log_dir ~/cs_5587 --dataset yelp

The log pattern we parse:
    Epoch  200 | loss=0.0235 pen=0.0005 | val R@20=0.0466 N@20=0.0335 | 4.5s
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+loss=([\d.eE+-]+)\s+pen=([\d.eE+-]+)\s+\|\s+"
    r"(val|test)\s+R@20=([\d.eE+-]+)\s+N@20=([\d.eE+-]+)"
)
FNAME_RE = re.compile(r"(?P<dataset>yelp|amazon)_temp_(?P<method>baseline|irm|vrex)(?:_(?:lam(?P<lam>[\d.]+)|rerun))?\.log")


def parse_log(path):
    epochs, r20, n20, loss, pen = [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                loss.append(float(m.group(2)))
                pen.append(float(m.group(3)))
                r20.append(float(m.group(5)))
                n20.append(float(m.group(6)))
    return {"epoch": np.array(epochs), "loss": np.array(loss), "pen": np.array(pen),
            "r20": np.array(r20), "n20": np.array(n20)}


def collect_runs(log_dir, dataset_prefix="yelp"):
    """Return list of (method, lam, curve_dict) for all matching logs."""
    runs = []
    for p in sorted(glob.glob(f"{log_dir}/{dataset_prefix}_temp_*.log")):
        name = Path(p).name
        m = FNAME_RE.match(name)
        if not m:
            continue
        method = m.group("method")
        lam_str = m.group("lam")
        lam = float(lam_str) if lam_str else None
        curve = parse_log(p)
        if len(curve["epoch"]) > 0:
            runs.append({"method": method, "lam": lam, "curve": curve, "name": name})
    return runs


def plot_training_curves(runs, metric="r20", title=""):
    methods = ["baseline", "irm", "vrex"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    cmap = {"irm": plt.cm.Blues, "vrex": plt.cm.Oranges}

    for ax_idx, focus_method in enumerate(["irm", "vrex"]):
        ax = axes[ax_idx]
        # plot baseline first as reference
        for run in runs:
            if run["method"] == "baseline":
                ax.plot(run["curve"]["epoch"], run["curve"][metric],
                        color="black", lw=1.5, label="baseline", alpha=0.9)

        # plot the focus method's runs in a color gradient by λ
        focus_runs = sorted([r for r in runs if r["method"] == focus_method], key=lambda r: r["lam"])
        lams = [r["lam"] for r in focus_runs]
        for i, run in enumerate(focus_runs):
            # normalize λ to [0.25, 0.95] for the colormap
            c = cmap[focus_method](0.25 + 0.70 * (i / max(1, len(focus_runs) - 1)))
            ax.plot(run["curve"]["epoch"], run["curve"][metric],
                    color=c, lw=1.4, label=f"λ={run['lam']}", alpha=0.9)

        ax.set_xlabel("epoch")
        ax.set_ylabel(f"val {metric.upper()}")
        ax.set_title(f"{title} — {focus_method.upper()}")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default=".", help="Directory containing *.log files")
    ap.add_argument("--dataset", default="yelp", choices=["yelp", "amazon"])
    ap.add_argument("--out_dir", default="paper/figures")
    args = ap.parse_args()

    runs = collect_runs(args.log_dir, args.dataset)
    print(f"found {len(runs)} runs:")
    for r in runs:
        print(f"  {r['name']}  ({len(r['curve']['epoch'])} eval points)")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fig = plot_training_curves(runs, metric="r20", title=args.dataset)
    fig.savefig(f"{args.out_dir}/training_curves_{args.dataset}.pdf")
    fig.savefig(f"{args.out_dir}/training_curves_{args.dataset}.png", dpi=200)
    print(f"saved {args.out_dir}/training_curves_{args.dataset}.pdf/.png")
