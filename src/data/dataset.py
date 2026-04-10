"""Dataset loading, temporal splitting, and graph construction."""

import sys
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from pathlib import Path
from typing import Tuple, Dict


def log(msg: str):
    print(msg, flush=True)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


class RecDataset:
    """Recommendation dataset with temporal train/val/test splits."""

    def __init__(self, name: str, val_ratio: float = 0.1, test_ratio: float = 0.2,
                 k_core: int = 10, allowlist: str = None,
                 split_mode: str = "temporal", split_files: str = None,
                 seed: int = 42):
        self.name = name
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        if split_mode == "files":
            assert split_files, "split_mode=files requires split_files prefix"
            log(f"Loading pre-split files (prefix={split_files})...")
            self.train_df, self.val_df, self.test_df, self.n_users, self.n_items = \
                self._load_split_files(split_files, val_ratio, seed)
            self.n_nodes = self.n_users + self.n_items
            log(f"  {self.n_users} users, {self.n_items} items, {self.n_nodes} nodes")
        else:
            suffix = f"_{allowlist}" if allowlist else ""
            cache_path = DATA_DIR / name / f"filtered_{k_core}core{suffix}.parquet"
            if k_core > 0 and cache_path.exists():
                log(f"Loading cached data from {cache_path}")
                lf = pl.read_parquet(cache_path)
            else:
                lf = self._load_interactions()
                if allowlist:
                    lf = self._apply_allowlist(lf, allowlist)
                if k_core > 0:
                    lf = self._k_core_filter(lf, k_core)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    lf.write_parquet(cache_path)
                    log(f"Cached filtered data to {cache_path}")

            log("Remapping IDs...")
            lf, self.n_users, self.n_items = self._remap_ids(lf)
            self.n_nodes = self.n_users + self.n_items
            log(f"  {self.n_users} users, {self.n_items} items, {self.n_nodes} nodes")

            if split_mode == "random":
                log(f"Random split (seed={seed})...")
                self.train_df, self.val_df, self.test_df = self._random_split(lf, seed)
            else:
                log("Temporal split...")
                self.train_df, self.val_df, self.test_df = self._temporal_split(lf)

        # Build adjacency for training graph
        log("Building adjacency matrix...")
        self.train_adj = self._build_adj(self.train_df)
        log("Normalizing adjacency...")
        self.norm_adj = self._normalize_adj(self.train_adj)

        # Precompute user interaction dicts for evaluation
        log("Building user-item dicts...")
        self.train_user_items = self._user_item_dict(self.train_df)
        self.val_user_items = self._user_item_dict(self.val_df)
        self.test_user_items = self._user_item_dict(self.test_df)

        log(
            f"Dataset: {name} | Users: {self.n_users} | Items: {self.n_items} | "
            f"Train: {len(self.train_df)} | Val: {len(self.val_df)} | Test: {len(self.test_df)}"
        )

    def _load_interactions(self) -> pl.DataFrame:
        """Load interactions as Polars DataFrame with columns [user, item, timestamp]."""
        if self.name == "ml-1m":
            path = DATA_DIR / "ml-1m" / "ratings.dat"
            df = pl.read_csv(
                path,
                separator="::",
                has_header=False,
                new_columns=["user", "item", "rating", "timestamp"],
            )
            df = df.select(["user", "item", "timestamp"])

        elif self.name == "amazon-books":
            path = DATA_DIR / "amazon-books" / "Books.csv"
            df = pl.read_csv(
                path,
                has_header=False,
                new_columns=["item", "user", "rating", "timestamp"],
                schema_overrides={"item": pl.Utf8, "user": pl.Utf8},
            )
            df = df.select(["user", "item", "timestamp"])

        elif self.name == "yelp2018":
            path = DATA_DIR / "yelp2018" / "yelp2018.inter"
            df = pl.read_csv(path, separator="\t", schema_overrides={"user_id:token": pl.Utf8, "item_id:token": pl.Utf8})
            df = df.rename({"user_id:token": "user", "item_id:token": "item", "timestamp:float": "timestamp"})
            df = df.select(["user", "item", "timestamp"])

        else:
            raise ValueError(f"Unknown dataset: {self.name}")

        return df

    def _load_split_files(self, prefix: str, val_ratio: float, seed: int):
        """Load NGCF-style train.txt/test.txt: each line `user_id item1 item2 ...` (already remapped)."""
        ds_dir = DATA_DIR / self.name

        def parse(path):
            users, items = [], []
            with open(path) as f:
                for line in f:
                    toks = line.strip().split()
                    if len(toks) < 2:
                        continue
                    u = int(toks[0])
                    for it in toks[1:]:
                        users.append(u)
                        items.append(int(it))
            return pl.DataFrame({
                "user": users, "item": items,
                "timestamp": [0] * len(users),
            }).with_columns([pl.col("user").cast(pl.Int64), pl.col("item").cast(pl.Int64)])

        train_full = parse(ds_dir / f"{prefix}_train.txt")
        test_df = parse(ds_dir / f"{prefix}_test.txt")

        n_users = max(train_full["user"].max(), test_df["user"].max()) + 1
        n_items = max(train_full["item"].max(), test_df["item"].max()) + 1

        # Random val holdout from train (per-user not strictly enforced; iid sample)
        rng = np.random.default_rng(seed)
        n = len(train_full)
        idx = rng.permutation(n)
        n_val = int(n * val_ratio)
        val_df = train_full[idx[:n_val].tolist()]
        train_df = train_full[idx[n_val:].tolist()]
        return train_df, val_df, test_df, n_users, n_items

    def _apply_allowlist(self, df: pl.DataFrame, allowlist: str) -> pl.DataFrame:
        """Filter to users/items listed in allowlist files (e.g. NGCF user_list.txt/item_list.txt)."""
        ds_dir = DATA_DIR / self.name
        users = pl.read_csv(ds_dir / f"{allowlist}_user_list.txt", separator=" ")["org_id"].cast(pl.Utf8)
        items = pl.read_csv(ds_dir / f"{allowlist}_item_list.txt", separator=" ")["org_id"].cast(pl.Utf8)
        before = len(df)
        df = df.filter(pl.col("user").is_in(users) & pl.col("item").is_in(items))
        log(f"Allowlist '{allowlist}': {before} -> {len(df)} interactions "
            f"({len(users)} allowed users, {len(items)} allowed items)")
        return df

    @staticmethod
    def _k_core_filter(df: pl.DataFrame, k: int) -> pl.DataFrame:
        """Iteratively filter users and items with fewer than k interactions."""
        prev_len = 0
        iteration = 0
        while len(df) != prev_len:
            prev_len = len(df)
            iteration += 1
            # Filter items with >= k interactions
            item_counts = df.group_by("item").len().filter(pl.col("len") >= k)
            df = df.join(item_counts.select("item"), on="item", how="inner")
            # Filter users with >= k interactions
            user_counts = df.group_by("user").len().filter(pl.col("len") >= k)
            df = df.join(user_counts.select("user"), on="user", how="inner")
            log(f"  k-core iteration {iteration}: {len(df)} interactions remaining")
        log(f"After {k}-core filtering: {len(df)} interactions")
        return df

    @staticmethod
    def _remap_ids(df: pl.DataFrame) -> Tuple[pl.DataFrame, int, int]:
        """Remap user and item IDs to contiguous 0-indexed integers."""
        unique_users = df["user"].unique().sort()
        unique_items = df["item"].unique().sort()

        user_map = pl.DataFrame({"user": unique_users, "user_id": range(len(unique_users))})
        item_map = pl.DataFrame({"item": unique_items, "item_id": range(len(unique_items))})

        df = df.join(user_map, on="user").join(item_map, on="item")
        df = df.drop(["user", "item"]).rename({"user_id": "user", "item_id": "item"})

        return df, len(unique_users), len(unique_items)

    def _temporal_split(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split by timestamp: 70% train, 10% val, 20% test."""
        df = df.sort("timestamp")
        n = len(df)
        train_end = int(n * (1 - self.val_ratio - self.test_ratio))
        val_end = int(n * (1 - self.test_ratio))
        return df[:train_end], df[train_end:val_end], df[val_end:]

    def _random_split(self, df: pl.DataFrame, seed: int) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Random iid split with same train/val/test ratios."""
        rng = np.random.default_rng(seed)
        n = len(df)
        idx = rng.permutation(n)
        train_end = int(n * (1 - self.val_ratio - self.test_ratio))
        val_end = int(n * (1 - self.test_ratio))
        return df[idx[:train_end].tolist()], df[idx[train_end:val_end].tolist()], df[idx[val_end:].tolist()]

    def _build_adj(self, df: pl.DataFrame) -> sp.csr_matrix:
        """Build bipartite adjacency matrix (n_users+n_items) x (n_users+n_items)."""
        users = df["user"].to_numpy()
        items = df["item"].to_numpy() + self.n_users

        rows = np.concatenate([users, items])
        cols = np.concatenate([items, users])
        data = np.ones(len(rows), dtype=np.float32)

        adj = sp.csr_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        return adj

    @staticmethod
    def _normalize_adj(adj: sp.csr_matrix) -> torch.sparse.FloatTensor:
        """Compute D^{-1/2} A D^{-1/2} and convert to torch sparse tensor."""
        adj = adj.tocoo()
        degrees = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = sp.diags(d_inv_sqrt)
        norm_adj = d_mat @ adj @ d_mat

        norm_adj = norm_adj.tocoo()
        indices = torch.LongTensor(np.stack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    @staticmethod
    def _user_item_dict(df: pl.DataFrame) -> Dict[int, set]:
        """Build {user: set(items)} dict."""
        grouped = df.group_by("user").agg(pl.col("item"))
        d = {}
        for row in grouped.iter_rows():
            d[row[0]] = set(row[1])
        return d

    def get_train_interactions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (users, items) arrays for training."""
        return self.train_df["user"].to_numpy(), self.train_df["item"].to_numpy()

    def sample_bpr_triplets(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample BPR triplets: (user, pos_item, neg_item)."""
        users, pos_items = self.get_train_interactions()
        n = len(users)

        idx = np.random.randint(0, n, size=batch_size)
        batch_users = users[idx]
        batch_pos = pos_items[idx]

        # Sample negative items (not interacted by user)
        batch_neg = np.random.randint(0, self.n_items, size=batch_size)
        for i in range(batch_size):
            u = batch_users[i]
            while batch_neg[i] in self.train_user_items.get(u, set()):
                batch_neg[i] = np.random.randint(0, self.n_items)

        return (
            torch.LongTensor(batch_users),
            torch.LongTensor(batch_pos),
            torch.LongTensor(batch_neg),
        )

    def item_degrees(self, split: str = "train") -> np.ndarray:
        """Return item degree array for a given split."""
        df = {"train": self.train_df, "val": self.val_df, "test": self.test_df}[split]
        degrees = np.zeros(self.n_items, dtype=np.int64)
        for item in df["item"].to_numpy():
            degrees[item] += 1
        return degrees
