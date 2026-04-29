# Reproducibility & CLI Execution Guide

This guide provides step-by-step instructions to reproduce all experimental findings, hyperparameter sweeps, and figures presented in the paper **Invariant Learning for GNN Recommenders Under Temporal Popularity Shift**.

---

## 1. Environment Setup

### Prerequisites
- Python $\ge$ 3.10
- PyTorch $\ge$ 2.0
- CUDA support (optional, recommended for GPU acceleration)

```bash
# Clone the repository
git clone https://github.com/mehakjain26/invariant-gnn-recommendation.git
cd invariant-gnn-recommendation

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Preparation

Download raw datasets (Yelp2018, Amazon-Books, MovieLens-1M) and perform $k$-core filtering and temporal partitioning:

```bash
# Download raw dataset files
python src/data/download.py --dataset yelp2018
python src/data/download.py --dataset amazon_books
python src/data/download.py --dataset ml1m

# Perform dataset analysis and compute Spearman rank correlation
python analyze_datasets.py
```

---

## 3. Running Model Benchmarks

Execute baseline, IRM, and V-REx models using the provided shell scripts or Python CLI.

### 3.1 Baseline LightGCN
```bash
bash scripts/run_baseline.sh
```
Or run directly:
```bash
python src/train.py --config configs/yelp2018_paper_temporal.yaml --method baseline
```

### 3.2 Invariant Risk Minimization (IRM)
```bash
bash scripts/run_irm.sh
```
Sweep penalty weight $\lambda \in \{0.1, 1.0, 10.0, 100.0\}$:
```bash
python src/train.py --config configs/yelp2018_paper_temporal.yaml --method irm --penalty_weight 10.0
```

### 3.3 Variance Risk Extrapolation (V-REx)
```bash
bash scripts/run_vrex.sh
```
Sweep penalty weight $\lambda \in \{0.1, 1.0, 10.0, 100.0\}$:
```bash
python src/train.py --config configs/amazon_books_paper_temporal.yaml --method vrex --penalty_weight 1.0
```

### 3.4 Full Hyperparameter & Depth Ablation Sweep
```bash
bash scripts/run_ablation.sh
```

---

## 4. Plot & Figure Generation

Generate all figures featured in the research paper:

```bash
# Plot interaction distribution over time
python scripts/plot_interactions_over_time.py

# Plot degree distribution comparisons
python scripts/plot_degree_dist.py

# Plot popularity rank correlation scatter plots (Spearman rho)
python scripts/plot_popularity_shift.py

# Plot environment partitioning boundaries
python scripts/plot_env_partition.py

# Plot subgroup Recall@20 comparison (Overall vs Falling vs Rising)
python scripts/plot_subgroup_recall.py

# Plot training convergence curves
python scripts/plot_training_curves.py
```
Output figures will be saved in `paper/figures/` in both `.pdf` and `.png` formats.
