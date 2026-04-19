"""Training loop for LightGCN with optional IRM/VREx invariant penalties."""

import argparse
import json
import os
import sys
import time

# Ensure project root is in sys.path for relative package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import RecDataset
from src.data.env_partition import partition_environments, assign_triplet_envs, get_env_masks, print_env_stats
from src.models.lightgcn import LightGCN
from src.losses.bpr import bpr_loss
from src.losses.irm import irm_penalty
from src.losses.vrex import vrex_penalty
from src.evaluate import evaluate, evaluate_subgroups
from src.utils import set_seed, load_config, get_device, ensure_dir


def train(cfg: dict):
    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"Using device: {device}")

    # --- Data ---
    dataset = RecDataset(
        cfg["dataset"],
        k_core=cfg.get("k_core", 10),
        allowlist=cfg.get("allowlist"),
        split_mode=cfg.get("split_mode", "temporal"),
        split_files=cfg.get("split_files"),
        seed=cfg.get("seed", 42),
    )
    norm_adj = dataset.norm_adj.to(device)

    # Environment partition
    n_envs = cfg.get("n_envs", 3)
    train_degrees = dataset.item_degrees("train")
    item_env_ids = partition_environments(train_degrees, n_envs=n_envs)
    print(f"Environment partition (E={n_envs}):")
    print_env_stats(train_degrees, item_env_ids)

    # --- Model ---
    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embed_dim=cfg.get("embed_dim", 64),
        n_layers=cfg.get("n_layers", 3),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    # --- Config ---
    method = cfg.get("method", "baseline")  # baseline, irm, vrex
    penalty_weight = cfg.get("penalty_weight", 1.0)
    reg_weight = cfg.get("reg_weight", 1e-4)
    batch_size = cfg.get("batch_size", 2048)
    epochs = cfg.get("epochs", 500)
    eval_every = cfg.get("eval_every", 10)
    patience = cfg.get("patience", 50)
    k = cfg.get("k", 20)

    vanilla_bpr = cfg.get("vanilla_bpr", False)

    # --- Logging ---
    allow_tag = f"_{cfg['allowlist']}" if cfg.get("allowlist") else ""
    vanilla_tag = "_vanilla" if vanilla_bpr else ""
    split_mode = cfg.get("split_mode", "temporal")
    if split_mode == "files":
        split_tag = f"_files-{cfg.get('split_files')}"
    elif split_mode == "random":
        split_tag = "_random"
    else:
        split_tag = ""
    reg_tag = f"_reg{reg_weight:.0e}".replace("e-0", "e-")
    n_layers = cfg.get("n_layers", 3)
    run_name = f"{cfg['dataset']}{split_tag}{allow_tag}_{method}{vanilla_tag}_E{n_envs}_lam{penalty_weight}{reg_tag}"
    # Route by BPR type and layer count:
    #   vanilla runs     -> results/layer_{N}/
    #   non-vanilla runs -> results/layer_{N}_non_vanilla/
    layer_subdir = f"layer_{n_layers}" if vanilla_bpr else f"layer_{n_layers}_non_vanilla"
    log_dir = os.path.join("results", layer_subdir, run_name)
    ensure_dir(log_dir)
    writer = SummaryWriter(log_dir)

    # Save config
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # --- Training ---
    best_recall = 0.0
    patience_counter = 0
    n_batches = max(1, len(dataset.train_df) // batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_penalty = 0.0
        t0 = time.time()

        for _ in range(n_batches):
            users, pos_items, neg_items = dataset.sample_bpr_triplets(batch_size)
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            # Forward
            user_emb, item_emb = model(norm_adj)
            u_emb = user_emb[users]
            pos_emb = item_emb[pos_items]
            neg_emb = item_emb[neg_items]

            pos_scores = model.score(u_emb, pos_emb)
            neg_scores = model.score(u_emb, neg_emb)

            # Per-environment masks
            triplet_envs = assign_triplet_envs(pos_items, item_env_ids).to(device)
            env_masks = get_env_masks(triplet_envs, n_envs)

            # BPR aggregation: vanilla = single mean over all triplets (matches LightGCN paper)
            # default = sum of per-env means (env-balanced, needed for IRM/VREx)
            if vanilla_bpr:
                total_bpr = bpr_loss(pos_scores, neg_scores)
            else:
                total_bpr = torch.tensor(0.0, device=device)
                for mask in env_masks:
                    total_bpr = total_bpr + bpr_loss(pos_scores, neg_scores, mask)

            # Invariant penalty
            if method == "irm":
                pen = irm_penalty(pos_scores, neg_scores, env_masks)
            elif method == "vrex":
                pen = vrex_penalty(pos_scores, neg_scores, env_masks)
            else:
                pen = torch.tensor(0.0, device=device)

            reg = reg_weight * model.l2_reg(users, pos_items, neg_items)
            loss = total_bpr + penalty_weight * pen + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += total_bpr.item()
            epoch_penalty += pen.item()

        epoch_loss /= n_batches
        epoch_penalty /= n_batches
        elapsed = time.time() - t0

        writer.add_scalar("train/bpr_loss", epoch_loss, epoch)
        writer.add_scalar("train/penalty", epoch_penalty, epoch)

        if epoch % eval_every == 0:
            # Evaluate on monitor split (val by default, test for paper-repro)
            monitor_split = cfg.get("early_stop_on", "val")
            monitor_items = dataset.test_user_items if monitor_split == "test" else dataset.val_user_items
            mon_metrics = evaluate(
                model, norm_adj,
                dataset.train_user_items, monitor_items,
                k=k,
            )
            recall = mon_metrics[f"recall@{k}"]
            ndcg = mon_metrics[f"ndcg@{k}"]
            writer.add_scalar(f"{monitor_split}/recall", recall, epoch)
            writer.add_scalar(f"{monitor_split}/ndcg", ndcg, epoch)

            print(
                f"Epoch {epoch:4d} | loss={epoch_loss:.4f} pen={epoch_penalty:.4f} | "
                f"{monitor_split} R@{k}={recall:.4f} N@{k}={ndcg:.4f} | {elapsed:.1f}s"
            )

            if recall > best_recall:
                best_recall = recall
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pt"))
            else:
                patience_counter += eval_every
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    # --- Final test evaluation ---
    model.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pt"), weights_only=True))
    test_metrics = evaluate(
        model, norm_adj,
        dataset.train_user_items, dataset.test_user_items,
        k=k,
    )
    print(f"\nTest results: {test_metrics}")

    # Subgroup analysis
    test_degrees = dataset.item_degrees("test")
    subgroup_results = evaluate_subgroups(
        model, norm_adj,
        dataset.train_user_items, dataset.test_user_items,
        train_degrees, test_degrees,
        k=k,
    )
    print(f"Subgroup results: {json.dumps(subgroup_results, indent=2, default=str)}")

    # Save results
    all_results = {"test": test_metrics, "subgroups": subgroup_results, "config": cfg}
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    writer.close()
    print(f"Results saved to {log_dir}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    # Allow CLI overrides
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--penalty_weight", type=float, default=None)
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--vanilla_bpr", action="store_true",
                        help="Use single mean BPR (matches standard LightGCN) instead of sum of per-env means")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    for key in ["method", "penalty_weight", "n_envs", "n_layers", "dataset"]:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    if args.vanilla_bpr:
        cfg["vanilla_bpr"] = True

    train(cfg)
