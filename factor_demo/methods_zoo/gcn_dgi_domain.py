"""Method 1: GCN-DGI Spatial Domain Detector (SpaceFlow / CCST / GraphST family).

Inductive Biases:
- Multi-hop unpruned spatial GCN smoothing (`n_hops=2` over spatial k-NN graph).
- Deep Graph Infomax (DGI) contrastive learning objective (maximizing mutual information
  between local patch embeddings and global tissue summary vs. feature-permuted negative graph).
- Default auxiliary pipeline: HVG selection (`hvg`) + Gaussian Mixture clustering (`gmm`).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from factor_demo.methods_zoo.base_pipeline import (
    MethodRunOutput,
    apply_graph_arm_hook,
    build_spatial_knn_graph,
    cluster_embedding,
    compute_effective_rank,
    preprocess_expression,
    unsupervised_silhouette_proxy,
)

METHOD_METADATA = {
    "method_id": "GCN_DGI",
    "display_name": "GCN-DGI (SpaceFlow/GraphST-style)",
    "family": "GNN_Contrastive",
}


def _train_dgi_encoder(
    features: np.ndarray,
    adj: np.ndarray,
    hidden_dim: int = 24,
    n_hops: int = 2,
    epochs: int = 45,
    lr: float = 0.03,
    spatial_reg_weight: float = 0.25,
    seed: int = 0,
) -> Tuple[np.ndarray, int, bool, float, float]:
    """Train a 2-hop GCN encoder with DGI contrastive + spatial smoothness objective."""
    rng = np.random.default_rng(seed)
    n_nodes, in_dim = features.shape

    # Pre-compute multi-hop propagated feature operator A^n_hops X (exact GCN message passing)
    prop_adj = np.eye(n_nodes, dtype=np.float32)
    for _ in range(n_hops):
        prop_adj = prop_adj @ adj

    pos_agg = prop_adj @ features
    W = rng.normal(0.0, np.sqrt(2.0 / (in_dim + hidden_dim)), size=(in_dim, hidden_dim)).astype(np.float32)
    b = np.zeros((1, hidden_dim), dtype=np.float32)
    disc_M = np.eye(hidden_dim, dtype=np.float32)
    param_count = int(W.size + b.size + disc_M.size)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    initial_loss = 1.0
    final_loss = 1.0
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)

    for ep in range(1, epochs + 1):
        perm = rng.permutation(n_nodes)
        neg_agg = prop_adj @ features[perm]

        h_pos = np.tanh(pos_agg @ W + b)
        h_neg = np.tanh(neg_agg @ W + b)
        summary = _sigmoid(np.mean(h_pos, axis=0, keepdims=True))  # (1, hidden_dim)

        proj = (summary @ disc_M).T  # (hidden_dim, 1)
        logits_pos = h_pos @ proj
        logits_neg = h_neg @ proj

        prob_pos = _sigmoid(logits_pos)
        prob_neg = _sigmoid(logits_neg)

        bce_loss = -float(np.mean(np.log(prob_pos + 1e-7) + np.log(1.0 - prob_neg + 1e-7)))
        # Spatial smoothness regularization on adjacent nodes in `adj`
        diff_h = h_pos - adj @ h_pos
        sp_loss = float(np.mean(diff_h ** 2)) * spatial_reg_weight
        loss = bce_loss + sp_loss

        if ep == 1:
            initial_loss = max(loss, 1e-6)
        final_loss = loss

        # Analytical gradient w.r.t. W
        d_h_pos = ((prob_pos - 1.0) @ proj.T) / n_nodes + (2.0 * spatial_reg_weight / (n_nodes * hidden_dim)) * (
            diff_h - adj.T @ diff_h
        )
        d_h_neg = (prob_neg @ proj.T) / n_nodes

        dz_pos = d_h_pos * (1.0 - h_pos ** 2)
        dz_neg = d_h_neg * (1.0 - h_neg ** 2)
        grad_W = pos_agg.T @ dz_pos + neg_agg.T @ dz_neg + 1e-4 * W
        grad_b = np.sum(dz_pos + dz_neg, axis=0, keepdims=True)

        # Adam update
        m_W = 0.9 * m_W + 0.1 * grad_W
        v_W = 0.999 * v_W + 0.001 * (grad_W ** 2)
        m_hat = m_W / (1.0 - 0.9 ** ep)
        v_hat = v_W / (1.0 - 0.999 ** ep)
        W -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        b -= lr * 0.5 * grad_b

    emb = np.tanh(pos_agg @ W + b)
    converged = bool(np.isfinite(final_loss) and final_loss <= initial_loss * 1.15)
    drop_ratio = float(final_loss / initial_loss)
    return emb, param_count, converged, float(final_loss), drop_ratio


def run_gcn_dgi(
    counts: np.ndarray,
    coords: np.ndarray,
    n_domains: int,
    arm: str = "orig",
    preprocess_override: Optional[str] = None,
    cluster_override: Optional[str] = None,
    tuning_protocol: str = "fixed",
    null_seed: int = 0,
    train_seed: int = 0,
) -> MethodRunOutput:
    """Execute GCN-DGI spatial domain pipeline with optional counterfactual hooks."""
    prep_mode = preprocess_override or "hvg"
    clust_mode = cluster_override or "gmm"

    feats = preprocess_expression(counts, mode=prep_mode, n_hvgs=110, n_pcs=20, seed=train_seed)

    if tuning_protocol == "fixed":
        candidate_configs = [{"k_neighbors": 10, "n_hops": 2, "spatial_reg_weight": 0.30}]
    else:
        # Matched-budget 3-point hyperparameter grid search using unsupervised criterion
        candidate_configs = [
            {"k_neighbors": 6, "n_hops": 1, "spatial_reg_weight": 0.10},
            {"k_neighbors": 10, "n_hops": 2, "spatial_reg_weight": 0.30},
            {"k_neighbors": 14, "n_hops": 2, "spatial_reg_weight": 0.50},
        ]

    best_score = -1e12
    best_output: Optional[MethodRunOutput] = None

    for cfg in candidate_configs:
        adj_raw = build_spatial_knn_graph(coords, k_neighbors=int(cfg["k_neighbors"]), include_self=True)
        adj_used = adj_raw if arm == "orig" else apply_graph_arm_hook(adj_raw, arm=arm, null_seed=null_seed)

        emb, n_params, conv, f_loss, drop = _train_dgi_encoder(
            features=feats,
            adj=adj_used,
            hidden_dim=24,
            n_hops=int(cfg["n_hops"]),
            spatial_reg_weight=float(cfg["spatial_reg_weight"]),
            seed=train_seed,
        )
        preds = cluster_embedding(emb, n_domains=n_domains, cluster_head=clust_mode, seed=train_seed)
        score = unsupervised_silhouette_proxy(emb, preds)
        if score > best_score or best_output is None:
            best_score = score
            best_output = MethodRunOutput(
                pred_labels=preds,
                embedding=emb,
                param_count=n_params,
                converged=conv,
                final_loss=f_loss,
                loss_drop_ratio=drop,
                effective_rank=compute_effective_rank(emb),
                hyperparams_used={**cfg, "preprocess": prep_mode, "cluster_head": clust_mode, "arm": arm},
            )

    assert best_output is not None
    return best_output
