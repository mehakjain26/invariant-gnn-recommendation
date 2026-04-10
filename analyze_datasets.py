import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import os
import polars as pl

# Try to import RecDataset from the local src
try:
    from src.data.dataset import RecDataset
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: Could not import RecDataset. {e}")
    exit(1)

def analyze_dataset(name):
    # Path check
    data_path = Path("data") / "raw" / name
    if not data_path.exists():
        print(f"\n[!] Dataset {name} not found at {data_path}")
        return

    print(f"\n--- Fingerprint: {name.upper()} ---")
    
    # Configure loading based on dataset
    kwargs = {}
    if name in ["amazon-books", "yelp2018"]:
        kwargs = {"split_mode": "files", "split_files": "ngcf"}
    
    # 1. Load Dataset
    try:
        ds = RecDataset(name, **kwargs)
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return
    
    n_users = ds.n_users
    n_items = ds.n_items
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)
    total_interactions = n_train + n_test + len(ds.val_df)

    # 2. Density & Sparsity
    possible_interactions = n_users * n_items
    density = total_interactions / possible_interactions
    sparsity = 1.0 - density

    # 3. Popularity Drift (Spearman Rho)
    train_degrees = ds.item_degrees("train")
    test_degrees = ds.item_degrees("test")
    
    # Ensure items exist in degrees array
    drift_rho, _ = spearmanr(train_degrees, test_degrees)

    # 4. Gini Coefficient (Popularity Skewness)
    def gini(x):
        sorted_x = np.sort(x)
        n = len(x)
        idx = np.arange(1, n + 1)
        return (np.sum((2 * idx - n - 1) * sorted_x)) / (n * np.sum(sorted_x))
    
    popularity_gini = gini(train_degrees)

    # 5. Summary Report
    print(f"Size: {n_users} users, {n_items} items")
    print(f"Interactions: {total_interactions:,} total")
    print(f"Sparsity: {sparsity * 100:.4f}%")
    print(f"Density: {density * 100:.6f}%")
    print(f"Popularity Drift (Rho): {drift_rho:.4f}")
    print(f"Gini (Skewness): {popularity_gini:.4f}")
    
    # 6. Gradient Stability Check
    if total_interactions < 100_000:
        print("Gradient Warning: Small data. Gradients might be very 'noisy' for IRM.")
    elif popularity_gini > 0.8:
        print("Skew Warning: High popularity bias. IRM/VRex will likely show significant gains.")
    else:
        print("Status: Stable enough for standard IRM training.")

if __name__ == "__main__":
    datasets = ["ml-1m", "amazon-books", "yelp2018"]
    for d in datasets:
        analyze_dataset(d)
