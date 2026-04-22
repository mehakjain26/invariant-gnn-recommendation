import matplotlib.pyplot as plt
import numpy as np
from src.data.dataset import RecDataset
from pathlib import Path

def compare_degree_distributions(datasets=["yelp2018", "amazon-books"]):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = {"yelp2018": "tab:orange", "amazon-books": "tab:blue"}
    
    for ds_name in datasets:
        print(f"Loading {ds_name}...")
        ds = RecDataset(ds_name, k_core=0, allowlist="ngcf", split_mode="files", split_files="ngcf")
        deg = ds.item_degrees("train")
        deg = np.sort(deg[deg > 0])
        
        # Calculate CDF
        cdf = np.arange(len(deg)) / float(len(deg))
        
        ax.plot(deg, cdf, label=f"{ds_name} (median={np.median(deg):.1f})", 
                color=colors[ds_name], linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("Item Degree (log)")
    ax.set_ylabel("CDF (Percentage of Items)")
    ax.set_title("Cumulative Degree Distribution: Yelp vs. Amazon")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    save_path = "paper/figures/compare_degree_dist.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved to {save_path}")
    return save_path

if __name__ == "__main__":
    compare_degree_distributions()
