"""Method 3: Neighborhood Feature Concatenation (BANKSY / CellCharter family).

Inductive Biases:
- Does NOT overwrite cell-intrinsic expression via multi-hop smoothing!
- Instead concatenates scaled cell-intrinsic features `[sqrt(1 - lam) * X_self]` with
  neighborhood context `[sqrt(lam) * A_nbr @ X_self]` in a joint product space, preserving
  sharp boundaries and thin/infiltrated microenvironments.
- Default auxiliary pipeline: PCA (`pca`) + KMeans clustering (`kmeans`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from factor_demo.methods_zoo.base_pipeline import (
    MethodRunOutput,
    apply_graph_arm_hook,
    build_spatial_knn_graph,
    cluster_embedding_with_status,
    compute_effective_rank,
    preprocess_expression,
    standardize_branch,
    unsupervised_silhouette_proxy,
)

METHOD_METADATA = {
    "method_id": "BANKSY_Concat",
    "display_name": "BANKSY-Style Concatenation",
    "family": "Feature_Concatenation",
}


def run_banksy_concat(
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
    """Execute BANKSY-style self + neighborhood concatenation clustering."""
    prep_mode = preprocess_override or "pca"
    clust_mode = cluster_override or "kmeans"

    feats = preprocess_expression(counts, mode=prep_mode, n_hvgs=110, n_pcs=20, seed=train_seed)

    if tuning_protocol == "fixed":
        candidate_configs = [{"k_neighbors": 8, "lambda_nbr": 0.35}]
    else:
        candidate_configs = [
            {"k_neighbors": 6, "lambda_nbr": 0.20},
            {"k_neighbors": 8, "lambda_nbr": 0.35},
            {"k_neighbors": 12, "lambda_nbr": 0.55},
        ]

    best_score = -1e12
    best_output: Optional[MethodRunOutput] = None

    for cfg in candidate_configs:
        # Exclude self-loop in raw neighbor context so concatenation separates self vs. environment
        adj_raw = build_spatial_knn_graph(coords, k_neighbors=int(cfg["k_neighbors"]), include_self=False)
        adj_used = adj_raw if arm == "orig" else apply_graph_arm_hook(adj_raw, arm=arm, null_seed=null_seed)

        lam = float(cfg["lambda_nbr"])
        # Standardize self and neighbor branches FIRST, then apply relative weights (following BANKSY)!
        z_self_std = standardize_branch(feats)
        z_nbr_std = standardize_branch(adj_used @ feats)
        emb = np.concatenate(
            [np.sqrt(1.0 - lam) * z_self_std, np.sqrt(lam) * z_nbr_std],
            axis=1,
        ).astype(np.float32)

        preds, clust_conv = cluster_embedding_with_status(
            emb, n_domains=n_domains, cluster_head=clust_mode, seed=train_seed, preserve_relative_weights=True
        )
        score = unsupervised_silhouette_proxy(emb, preds)
        if score > best_score or best_output is None:
            best_score = score
            best_output = MethodRunOutput(
                pred_labels=preds,
                embedding=emb,
                param_count=int(emb.shape[1]),
                converged=clust_conv,
                final_loss=0.0,
                loss_drop_ratio=1.0,
                effective_rank=compute_effective_rank(emb),
                hyperparams_used={**cfg, "preprocess": prep_mode, "cluster_head": clust_mode, "arm": arm},
            )

    assert best_output is not None
    return best_output
