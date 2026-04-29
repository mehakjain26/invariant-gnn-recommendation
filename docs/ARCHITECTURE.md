# Architectural Framework & Technical Specifications

This document details the theoretical foundation, formal mathematical models, and loss objective formulations implemented in **Invariant Learning for GNN Recommenders Under Temporal Popularity Shift**.

---

## 1. Backbone GNN: LightGCN

Standard Graph Convolutional Networks (GCNs) incorporate heavy non-linear activations and weight transformation matrices. As shown by He et al. (2020), these operations degrade performance in collaborative filtering. **LightGCN** simplifies the graph convolution operation by retaining only symmetric-normalized neighborhood aggregation.

### 1.1 Layer-wise Message Passing

Let $\mathcal{G} = (\mathcal{U}, \mathcal{V}, \mathcal{E})$ be the bipartite user-item interaction graph, where $|\mathcal{U}|$ is the number of users and $|\mathcal{V}|$ is the number of items. 

The initial embedding table $E^{(0)} \in \mathbb{R}^{(|\mathcal{U}| + |\mathcal{V}|) \times d}$ represents the only learnable parameters in the model. The propagation rule at layer $k+1$ is defined as:

$$e_{u}^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u| \cdot |\mathcal{N}_i|}} e_{i}^{(k)}$$

$$e_{i}^{(k+1)} = \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i| \cdot |\mathcal{N}_u|}} e_{u}^{(k)}$$

where $\mathcal{N}_u$ denotes the set of items interacted with by user $u$, and $\mathcal{N}_i$ denotes the set of users who interacted with item $i$.

### 1.2 Multi-layer Readout

The final representation $e^{\ast}$ is computed as the mean pooling across all $L$ convolution layers:

$$e_u^{\ast} = \frac{1}{L+1} \sum_{k=0}^L e_u^{(k)}, \quad e_i^{\ast} = \frac{1}{L+1} \sum_{k=0}^L e_i^{(k)}$$

The affinity prediction score $\hat{y}_{ui}$ for a user-item pair $(u, i)$ is computed via the dot product:

$$\hat{y}_{ui} = \langle e_u^{\ast}, e_i^{\ast} \rangle$$

---

## 2. Structural Graph Environment Partitioning

To break the reliance on spurious popularity correlations, the global training graph $\mathcal{G}\_{\text{train}}$ is partitioned into $E$ discrete structural pseudo-environments:

$$\mathcal{E}_{\text{tr}} = \{e_1, e_2, \dots, e_E\}$$

based on training node degrees $\text{deg}(i)$.

For $E = 3$ environments:
- **$e_1$ (Popular Environment)**: Items in the top 20th degree percentile.
- **$e_2$ (Average Environment)**: Items in the 20th–80th degree percentile.
- **$e_3$ (Tail / Unknown Environment)**: Items in the bottom 20th degree percentile.

Each environment $e$ induces a specific subgraph:

$$\mathcal{G}_e \subset \mathcal{G}_{\text{train}}$$

containing its respective item subset and all interacting users.

---

## 3. Loss Formulations & Aggregation Modes

### 3.1 Base Risk: Bayesian Personalized Ranking (BPR)

For any given environment $e$, the per-environment BPR loss is computed over triplets $(u, i, j)$ where user $u$ interacted with positive item $i \in e$, and $j$ is a randomly sampled negative item:

$$R^e(\Phi, w) = \mathbb{E}_{(u, i, j) \in e} \left[ -\log \sigma (\hat{y}_{ui} - \hat{y}_{uj}) \right] + \lambda_2 \| E^{(0)} \|_2^2$$

### 3.2 Loss Aggregation Modes

We investigate two distinct loss aggregation methods across environments:

#### 1. Vanilla BPR Aggregation
The objective is computed as a single stochastic mean over all sampled triplets:

$$\mathcal{L}_{\text{Vanilla}} = \frac{1}{|B|} \sum_{(u,i,j) \in B} \ell_{\text{BPR}}(u,i,j)$$

*Consequence*: Denser environments ($e_1, e_2$) dominate batch gradients, leaving tail items under-optimized.

#### 2. Balanced BPR Aggregation
Per-environment losses are calculated independently and summed equally:

$$\mathcal{L}_{\text{Balanced}} = \sum_{e=1}^E R^e(\Phi, w)$$

*Consequence*: Forces equal gradient influence from ultra-sparse tail environments, significantly improving zero-shot discovery on nascent ("Rising") items.

---

## 4. Invariant Penalty Functions

The overall training objective combines empirical risk with a causal penalty $\Omega$:

$$\min_{\Phi, w} \sum_{e=1}^E R^e(\Phi, w) + \lambda \cdot \Omega\left(\{R^e\}_{e=1}^E\right)$$

where $\lambda$ controls penalty strength.

### 4.1 Invariant Risk Minimization (IRM)

IRM enforces feature alignment by requiring a dummy scalar multiplier $w = 1.0$ to be simultaneously optimal across all training environments:

$$\Omega_{\text{IRM}} = \sum_{e=1}^E \| \nabla_{w=1.0} R^e(w \cdot \Phi) \|^2$$

*Operational Behavior*: On hyper-sparse graphs (Amazon-Books, 0.06% density), chaotic drift makes finding an intersecting gradient space impossible, leading to constraint collapse.

### 4.2 Variance Risk Extrapolation (V-REx)

V-REx enforces macroscopic loss alignment by directly penalizing the variance of risks across environments:

$$\Omega_{\text{V-REx}} = \text{Var}\left( \{ R^1(\Phi, w), R^2(\Phi, w), \dots, R^E(\Phi, w) \} \right)$$

*Operational Behavior*: Because loss variance is a macroscopic statistic, V-REx is highly resilient to microscopic batch noise and provides stable regularization on ultra-sparse bipartite graphs.
