"""Method 2: Expression-Gated Graph Attention Autoencoder (STAGATE-style).

Inductive Biases:
- Adaptive attention-gated spatial graph: spatial edges are weighted by transcriptional
  cosine similarity so dissimilar boundary cells receive attenuated spatial smoothing.
- Reconstruction autoencoder objective (minimizing MSE reconstruction loss of input features).
- Default auxiliary pipeline: PCA dimension reduction (`pca`) + GMM clustering (`gmm`).
"""

from __future__ import annotations

from typing import Optional, Tuple

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
    "method_id": "GAT_AE",
    "display_name": "GAT-AutoEncoder (STAGATE-style)",
    "family": "GNN_Autoencoder",
}


def _compute_attention_gated_graph(
    features: np.ndarray,
    adj_base: np.ndarray,
    gate_temperature: float = 2.0,
) -> np.ndarray:
    """Gate spatial adjacency edges using transcriptional cosine similarity attention."""
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
    normed_f = features / norms
    sim = normed_f @ normed_f.T  # cosine similarity in [-1, 1]

    mask = (adj_base > 0).astype(np.float32)
    att_weights = np.exp(gate_temperature * sim) * mask
    row_sum = np.sum(att_weights, axis=1, keepdims=True)
    return att_weights / np.maximum(row_sum, 1e-6)


def _train_gat_autoencoder(
    features: np.ndarray,
    adj: np.ndarray,
    gate_temperature: float = 2.0,
    latent_dim: int = 20,
    epochs: int = 50,
    lr: float = 0.025,
    seed: int = 0,
) -> Tuple[np.ndarray, int, bool, float, float]:
    """Train attention-gated spatial autoencoder via self-reconstruction loss."""
    rng = np.random.default_rng(seed)
    n_nodes, in_dim = features.shape
    d_lat = min(latent_dim, in_dim)

    att_adj = _compute_attention_gated_graph(features, adj_base=adj, gate_temperature=gate_temperature)
    gated_input = 0.55 * features + 0.45 * (att_adj @ features)

    W_enc = rng.normal(0.0, np.sqrt(2.0 / (in_dim + d_lat)), size=(in_dim, d_lat)).astype(np.float32)
    W_dec = W_enc.T.copy()
    param_count = int(W_enc.size + W_dec.size)

    init_loss = 1.0
    final_loss = 1.0
    for ep in range(1, epochs + 1):
        z = np.tanh(gated_input @ W_enc)
        z_smooth = 0.65 * z + 0.35 * (att_adj @ z)
        recon = z_smooth @ W_dec
        err = recon - features
        loss = float(np.mean(err ** 2))
        if ep == 1:
            init_loss = max(loss, 1e-6)
        final_loss = loss

        grad_W_dec = (2.0 / (n_nodes * in_dim)) * (z_smooth.T @ err) + 1e-4 * W_dec
        d_z_smooth = (2.0 / (n_nodes * in_dim)) * (err @ W_dec.T)
        d_z = 0.65 * d_z_smooth + 0.35 * (att_adj.T @ d_z_smooth)
        dz_pre = d_z * (1.0 - z ** 2)
        grad_W_enc = gated_input.T @ dz_pre + 1e-4 * W_enc

        W_enc -= lr * grad_W_enc
        W_dec -= lr * grad_W_dec

    emb = 0.65 * np.tanh(gated_input @ W_enc) + 0.35 * (att_adj @ np.tanh(gated_input @ W_enc))
    converged = bool(np.isfinite(final_loss) and final_loss <= init_loss * 1.05)
    return emb, param_count, converged, final_loss, float(final_loss / init_loss)


def run_gat_ae(
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
    """Execute GAT-AutoEncoder pipeline with counterfactual hooks."""
    prep_mode = preprocess_override or "pca"
    clust_mode = cluster_override or "gmm"

    feats = preprocess_expression(counts, mode=prep_mode, n_hvgs=110, n_pcs=20, seed=train_seed)

    if tuning_protocol == "fixed":
        candidate_configs = [{"k_neighbors": 8, "gate_temperature": 2.2}]
    else:
        candidate_configs = [
            {"k_neighbors": 6, "gate_temperature": 3.2},
            {"k_neighbors": 8, "gate_temperature": 2.2},
            {"k_neighbors": 12, "gate_temperature": 1.0},
        ]

    best_score = -1e12
    best_output: Optional[MethodRunOutput] = None

    for cfg in candidate_configs:
        adj_raw = build_spatial_knn_graph(coords, k_neighbors=int(cfg["k_neighbors"]), include_self=True)
        adj_used = adj_raw if arm == "orig" else apply_graph_arm_hook(adj_raw, arm=arm, null_seed=null_seed)

        emb, n_params, conv, f_loss, drop = _train_gat_autoencoder(
            features=feats,
            adj=adj_used,
            gate_temperature=float(cfg["gate_temperature"]),
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
