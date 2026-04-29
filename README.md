# Invariant Learning for GNN Recommenders Under Temporal Popularity Shift

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Graph ML](https://img.shields.io/badge/Domain-Graph%20ML%20%26%20Causal%20AI-6f42c1?style=for-the-badge)](docs/ARCHITECTURE.md)
[![Course](https://img.shields.io/badge/Course-CS%20587%3A%20Deep%20Learning-CEB888?style=for-the-badge)](docs/ARCHITECTURE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![CI Build](https://img.shields.io/badge/CI-Passing-brightgreen?style=for-the-badge&logo=githubactions&logoColor=white)](.github/workflows/ci.yml)

**Author**: **Mehak Jain** (Purdue University | `jain1002@purdue.edu`)  
**Project Context**: Developed as a Research Project for **CS 587: Deep Learning** (Spring 2026, Purdue University).

---

## 📌 Executive Summary

Standard recommender systems (e.g., e-commerce, streaming platforms) are typically evaluated on **random data splits**, which artificially inflate accuracy by leaking future user trends into training data. When deployed in production over time, these systems encounter **temporal popularity shift**—causing them to recommend only currently viral items while failing on high-quality niche or formerly popular items.

This project addresses this critical production ML challenge by applying **Causal Inference & Invariant Machine Learning** (**Invariant Risk Minimization - IRM** and **Variance Risk Extrapolation - V-REx**) to **LightGCN Graph Neural Networks**. Evaluated across commercial datasets (**Amazon-Books**, **Yelp2018**, and **MovieLens-1M**), this repository establishes how to prevent models from relying on transient popularity shortcuts.

---

## ⚙️ How It Works: System Architecture & Workflow

The pipeline is structured into 5 recruiter-scannable modular stages:

1. **User-Item Graph Construction**:
   - Builds an interaction graph $\mathcal{G} = (\mathcal{U}, \mathcal{V}, \mathcal{E})$ from historical user engagement data and interaction timestamps.

2. **Graph Neural Network (LightGCN) Encoder**:
   - Uses LightGCN to propagate user preferences across graph connections without heavy parameter weight matrices.
   - Combines multi-layer graph convolutions into final user and item vector representations.

3. **Popularity-Based Environment Splitting**:
   - Automatically partitions items into 3 engagement tiers based on historical popularity percentiles:
     - **Popular ($e_1$)**: High-volume top items (top 20%).
     - **Average ($e_2$)**: Mid-tier items (middle 60%).
     - **Tail / Niche ($e_3$)**: Low-volume items (bottom 20%).

4. **Causal Regularization & Invariant Loss Functions**:
   - Evaluates Bayesian Personalized Ranking (BPR) recommendation loss independently per popularity tier.
   - Applies causal penalties scaled by weight $\lambda$:
     - **IRM (Invariant Risk Minimization)**: Forces model gradients to align across all popularity tiers to learn invariant preference features.
     - **V-REx (Variance Risk Extrapolation)**: Penalizes risk variance across popularity tiers to prevent the model from over-indexing on popular items at the expense of niche items.

5. **Real-World Out-of-Distribution (OOD) Evaluation**:
   - Evaluates performance chronologically on real future data rather than artificial random splits.
   - Specifically measures accuracy on:
     - **Falling Items**: Formerly viral items losing popularity (tests robustness against trend decay).
     - **Rising Items**: Brand-new items starting to go viral (tests cold-start discovery).

---

## 🚀 Key Research Findings & Contributions

1. **Boundary Conditions for Causal Recommendation**:
   - On hyper-sparse, dynamically chaotic graphs (**Amazon-Books**, 0.06% density, Spearman $\rho=0.24$), **IRM gradient-matching collapses completely**, degrading out-of-distribution Falling Recall by **16%**.
   - Conversely, **V-REx**'s macroscopic loss-alignment penalty remains resilient to microscopic batch noise, boosting subgroup robustness by **+19% on Amazon** and over **+77% on Yelp2018**.

2. **Base Loss Aggregation Trade-off (Vanilla vs. Balanced BPR)**:
   - **Balanced BPR** (summing equal per-environment means) forces the loss to attend to sparse tail environments, boosting zero-shot **Rising Item Recall by up to +150% on Yelp** and **+140% on Amazon**.

3. **GNN Architectural Depth as a Temporal Regularizer**:
   - Expanding the topological receptive field ($L=1 \to 4$ layers) natively improves temporal stability—increasing baseline out-of-distribution robustness by **+55% on Amazon** without requiring causal penalties.

4. **The Asymmetry of Discovery**:
   - Models are **20x–30x better** at retaining semantic embeddings for formerly popular "falling" items than predicting the zero-shot emergence of nascent "rising" items.

---

## 📊 Empirical Benchmarks

### 1. Dataset Topologies & Popularity Drift ($\rho$)

| Dataset | Users | Items | Interactions | Density | Sparsity | Spearman $\rho$ | Top-100 Overlap | Shift Regime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MovieLens-1M** | 6,040 | 3,260 | ~1.0M | 5.07% | 94.92% | 0.9495 | 91% | Stationary |
| **Yelp2018** | 31,668 | 38,048 | ~1.5M | 0.13% | 99.87% | 0.5123 | 48% | **Goldilocks Zone** |
| **Amazon-Books** | 52,643 | 91,599 | ~2.9M | **0.06%** | **99.93%** | **0.2464** | **23%** | **Chaotic Shift** |

### 2. Primary Invariant Performance (Recall@20 under Temporal Split)

| Method | Penalty ($\lambda$) | Amazon-Books (Overall) | Amazon-Books (Falling) | Amazon-Books (Rising) | Yelp2018 (Overall) | Yelp2018 (Falling) | Yelp2018 (Rising) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | -- | 0.0273 | 0.0148 | 0.0005 | 0.0321 | 0.0166 | 0.0002 |
| **V-REx** | 0.1 | 0.0273 | 0.0148 | 0.0005 | 0.0318 | 0.0203 (+22%) | 0.0002 |
| **V-REx** | 1.0 | 0.0269 | **0.0177 (+19%)** | 0.0006 | 0.0319 | 0.0184 (+11%) | 0.0002 |
| **V-REx** | 10.0 | 0.0265 | 0.0172 (+16%) | 0.0005 | 0.0317 | **0.0295 (+77%)** | 0.0001 |
| **V-REx** | 100.0 | 0.0257 | 0.0148 (+0%) | **0.0007 (+40%)** | 0.0316 | 0.0276 (+66%) | 0.0001 |
| **IRM** | 0.1 | 0.0272 | 0.0148 | 0.0005 | 0.0320 | 0.0147 (-11%) | 0.0002 |
| **IRM** | 1.0 | 0.0267 | 0.0124 (-16%) | 0.0006 | 0.0317 | 0.0249 (+50%) | 0.0001 |
| **IRM** | 10.0 | 0.0266 | 0.0130 (-12%) | 0.0006 | 0.0315 | **0.0285 (+71%)** | **0.0003 (+50%)** |
| **IRM** | 100.0 | 0.0188 *(Collapse)* | 0.0137 (-7%) | 0.0002 | 0.0210 | 0.0018 (-89%) | 0.0000 |

### 3. Impact of BPR Loss Balancing on Recall@20 ($\lambda=10.0$)

| Dataset | Method | Vanilla BPR (Falling) | Vanilla BPR (Rising) | Balanced BPR (Falling) | Balanced BPR (Rising) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Yelp2018** | **V-REx** | 0.0295 (+77%) | 0.0001 (-50%) | 0.0184 (+10%) | **0.0004 (+100%)** |
| **Yelp2018** | **IRM** | 0.0285 (+71%) | 0.0003 (+50%) | 0.0212 (+27%) | **0.0005 (+150%)** |
| **Amazon-Books** | **V-REx** | 0.0172 (+16%) | 0.0005 (+0%) | 0.0148 (+0%) | **0.0012 (+140%)** |
| **Amazon-Books** | **IRM** | 0.0130 (-12%) | 0.0006 (+20%) | 0.0148 (+0%) | **0.0010 (+100%)** |

---

## 🛠️ Quickstart & Reproduction Guide

### 1. Clone & Installation
```bash
git clone https://github.com/mehakjain26/invariant-gnn-recommendation.git
cd invariant-gnn-recommendation

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Processing
```bash
python src/data/download.py --dataset yelp2018
python analyze_datasets.py
```

### 3. Training & Evaluation
```bash
# Run Baseline LightGCN
python src/train.py --config configs/yelp2018_paper_temporal.yaml --method baseline

# Run V-REx (lambda=10.0)
python src/train.py --config configs/yelp2018_paper_temporal.yaml --method vrex --penalty_weight 10.0

# Run IRM (lambda=10.0)
python src/train.py --config configs/yelp2018_paper_temporal.yaml --method irm --penalty_weight 10.0
```

### 4. Reproducing Paper Figures
```bash
python scripts/plot_popularity_shift.py
python scripts/plot_subgroup_recall.py
python scripts/plot_training_curves.py
```

---

## 📂 Repository Structure

```
.
├── configs/                       # YAML hyperparameter configurations
│   ├── amazon_books_paper_temporal.yaml
│   ├── base.yaml
│   ├── ml1m.yaml
│   └── yelp2018_paper_temporal.yaml
├── docs/                          # Deep-dive documentation
│   ├── ARCHITECTURE.md            # Mathematical formulations & loss equations
│   ├── DATASETS.md                # Dataset topology & OOD subgroup breakdown
│   └── REPRODUCIBILITY.md         # Comprehensive CLI reference guide
├── paper/                         # Paper LaTeX source & vector plots
│   └── figures/                   # Generated PNG/PDF publication figures
├── scripts/                       # Training sweeps & visualization tools
│   ├── run_ablation.sh            # Hyperparameter sweep launcher
│   ├── plot_popularity_shift.py   # Spearman rank correlation scatter plots
│   └── plot_subgroup_recall.py    # OOD subgroup Recall@20 bar plots
├── src/                           # Main Python source package
│   ├── data/                      # Data loaders & environment partitioning
│   ├── losses/                    # BPR, IRM, and V-REx penalty implementations
│   ├── models/                    # LightGCN graph encoder
│   ├── evaluate.py                # Subgroup evaluation routines
│   ├── train.py                   # Main training loop
│   └── utils.py                   # Helper utilities
├── analyze_datasets.py            # Spearman rho dataset analysis script
├── LICENSE                        # MIT License
└── requirements.txt               # Dependencies
```

---

## 📖 Deep-Dive Documentation

- [📐 Mathematical Architecture & Specifications](docs/ARCHITECTURE.md)
- [📊 Dataset Topologies & Temporal Shift](docs/DATASETS.md)
- [🔄 Full Reproducibility Manual](docs/REPRODUCIBILITY.md)

---

## 📜 Citation

If you find this work or codebase useful in your research, please cite:

```bibtex
@article{jain2026invariant,
  title={Invariant Learning for GNN Recommenders Under Temporal Popularity Shift},
  author={Jain, Mehak},
  note={Research Project for CS 587: Deep Learning (Spring 2026)},
  institution={Purdue University},
  year={2026}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).
