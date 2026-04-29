# Datasets & Popularity Shift Topologies

This document provides a breakdown of dataset statistics, temporal splitting protocols, Spearman rank correlation ($\rho$) drift metrics, and evaluation subgroup definitions.

---

## 1. Dataset Overview & Topological Statistics

We evaluate invariant GNN recommenders across three standard collaborative filtering benchmarks exhibiting contrasting density and drift characteristics.

| Dataset | Users | Items | Interactions | Density | Sparsity | Spearman $\rho$ | Top-100 Overlap | Drift Regime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MovieLens-1M** | 6,040 | 3,260 | ~1.0M | 5.07% | 94.92% | 0.9495 | 92% | **Stationary** |
| **Yelp2018** | 31,668 | 38,048 | ~1.5M | 0.13% | 99.87% | 0.5123 | 48% | **Goldilocks Zone** |
| **Amazon-Books** | 52,643 | 91,599 | ~2.9M | **0.06%** | **99.93%** | **0.2464** | **23%** | **Chaotic Shift** |

---

## 2. Temporal Splitting Protocol

Unlike traditional evaluation paradigms that use random data splitting (which artificially leaks future engagement trends), all datasets in this repository are split chronologically by interaction timestamp:

- **Training Set ($\mathcal{D}\_{\text{train}}$)**: First **70%** of interactions chronologically.
- **Validation Set ($\mathcal{D}\_{\text{val}}$)**: Next **10%** of interactions.
- **Test Set ($\mathcal{D}\_{\text{test}}$)**: Final **20%** of interactions.

### Measuring Popularity Shift ($\rho$)

Popularity shift is quantified using **Spearman's Rank Correlation ($\rho$)** between item interaction ranks in $\mathcal{D}\_{\text{train}}$ and $\mathcal{D}\_{\text{test}}$:

$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

- **MovieLens-1M ($\rho = 0.95$)**: Popularity rankings remain virtually static across time. Popularity-based heuristics remain effective.
- **Yelp2018 ($\rho = 0.51$)**: Popularity shifts significantly but systematically. This "Goldilocks Zone" offers sufficient structural signal for causal gradient matching.
- **Amazon-Books ($\rho = 0.24$)**: Violent, unpredictable rank shifts occur (only 23% Top-100 overlap). Microscopic gradient matching (IRM) fails, whereas macroscopic loss variance regularization (V-REx) thrives.

---

## 3. Out-of-Distribution Subgroup Definitions

To evaluate true out-of-distribution performance, the test evaluation set is partitioned into three key item subgroups based on historical vs. test interaction volume:

1. **Falling Items ($\mathcal{V}_{\text{falling}}$)**:
   - Items whose engagement drops significantly from training to test windows.
   - Represents items falling out of favor. Standard BPR baseline accuracy degrades heavily on this group.
2. **Rising Items ($\mathcal{V}_{\text{rising}}$)**:
   - Nascent items with sparse training interactions (1–5 interactions) that experience sudden viral growth in the test set.
   - Tests zero-shot discovery capabilities.
3. **Stable Items ($\mathcal{V}_{\text{stable}}$)**:
   - Items maintaining consistent popularity rank across both time windows.
