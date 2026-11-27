"""Method 4: Non-Spatial Expression Clustering + Post-hoc Majority-Vote Smoothing Baseline.

Follows Descoeudres et al. (2026) baseline design:
- Clusters cells purely by transcriptional embedding (`pca` + `kmeans`).
- Optionally applies post-hoc local spatial majority vote over `A` (controlled by `arm`:
  `orig`/`refac` = spatial majority vote; `self` = pure non-spatial clustering;
  `null` = majority vote over permuted null neighbors).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from factor_demo.methods_zoo.base_pipeline import (
    MethodRunOutput,
    apply_graph_arm_hook,
    build_spatial_knn_graph,
    cluster_embedding,
    compute_effective_rank,
    preprocess_expression,
)

METHOD_METADATA = {
    "method_id": "Expr_Smooth_Baseline",
    "display_name": "NonSpatial+Smooth Baseline",
    "family": "Posthoc_Baseline",
}


def _majority_vote_smooth(labels: np.ndarray, adj: np.ndarray, n_domains: int) -> np.ndarray:
    """Reassign label by weighted majority vote across rows of `adj`."""
    one_hot = np.eye(n_domains, dtype=np.float32)[labels]
    votes = adj @ one_hot
    return np.argmax(votes, axis=1).astype(np.int64)


def run_expr_baseline(
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
    """Execute transcriptional baseline with post-hoc spatial majority vote."""
    prep_mode = preprocess_override or "pca"
    clust_mode = cluster_override or "kmeans"

    feats = preprocess_expression(counts, mode=prep_mode, n_hvgs=110, n_pcs=20, seed=train_seed)
    raw_preds = cluster_embedding(feats, n_domains=n_domains, cluster_head=clust_mode, seed=train_seed)

    k_nbr = 6 if tuning_protocol == "fixed" else 8
    adj_raw = build_spatial_knn_graph(coords, k_neighbors=k_nbr, include_self=True)
    adj_used = adj_raw if arm == "orig" else apply_graph_arm_hook(adj_raw, arm=arm, null_seed=null_seed)

    preds = _majority_vote_smooth(raw_preds, adj=adj_used, n_domains=n_domains)
    return MethodRunOutput(
        pred_labels=preds,
        embedding=feats,
        param_count=int(feats.shape[1]),
        converged=True,
        final_loss=0.0,
        loss_drop_ratio=1.0,
        effective_rank=compute_effective_rank(feats),
        hyperparams_used={"k_neighbors": k_nbr, "preprocess": prep_mode, "cluster_head": clust_mode, "arm": arm},
    )
