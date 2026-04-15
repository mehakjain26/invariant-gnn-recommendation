"""LightGCN: Simplified Graph Convolution for Recommendation."""

import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """LightGCN encoder.

    Only learnable parameters are the initial embedding table E^(0).
    Message passing: E^(k) = D^{-1/2} A D^{-1/2} E^{(k-1)}
    Final embedding: mean of E^(0), ..., E^(K)
    """

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_users + n_items, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, norm_adj: torch.sparse.FloatTensor) -> tuple:
        """Run K-layer graph convolution.

        Args:
            norm_adj: Normalized adjacency matrix D^{-1/2} A D^{-1/2},
                      shape (n_users+n_items, n_users+n_items), sparse.

        Returns:
            user_emb: Final user embeddings, shape (n_users, embed_dim).
            item_emb: Final item embeddings, shape (n_items, embed_dim).
        """
        all_emb = self.embedding.weight  # (n_users + n_items, d)
        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)

        # Mean pooling over layers
        final_emb = torch.stack(embs, dim=0).mean(dim=0)

        user_emb = final_emb[: self.n_users]
        item_emb = final_emb[self.n_users :]
        return user_emb, item_emb

    def score(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Compute dot-product scores.

        Args:
            user_emb: (batch, d)
            item_emb: (batch, d)

        Returns:
            scores: (batch,)
        """
        return (user_emb * item_emb).sum(dim=-1)

    def l2_reg(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> torch.Tensor:
        """L2 regularization on initial embeddings for the batch."""
        user_emb_0 = self.embedding.weight[users]
        pos_emb_0 = self.embedding.weight[pos_items + self.n_users]
        neg_emb_0 = self.embedding.weight[neg_items + self.n_users]
        return (
            user_emb_0.norm(2).pow(2)
            + pos_emb_0.norm(2).pow(2)
            + neg_emb_0.norm(2).pow(2)
        ) / (2 * len(users))
