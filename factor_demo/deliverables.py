"""Downstream Actionable Deliverables synthesized from the FACTOR Framework (v2.2 Audited).

Incorporates all methodological fixes from review:
1. **Strict 3-Split Separation (`Exploration` -> `Confirmation` -> `External_Test`)**:
   Downstream tools are calibrated on `Exploration` (with design hypotheses verified on `Confirmation`)
   and evaluated strictly on untouched `External_Test` donors (`T01..T03`), tracking end-to-end
   wall-clock runtime (feature extraction + candidate runs + fusion).
2. **Branch Standardization Before Weighting**:
   Standardizes `Z_self` and `Z_nbr` individually to unit variance BEFORE applying `[sqrt(1-lam), sqrt(lam)]`
   and clusters with `preserve_relative_weights=True` so weights are never canceled by column z-scoring!
3. **4-Way Ablation inside `FactorAuditedChimera`**:
   Explicitly compares:
   - `"fixed_concat"` (BANKSY-like fixed `lam=0.35`, uniform kNN weights)
   - `"gated_only"` (fixed `lam=0.35` + cosine expression similarity edge gating)
   - `"adaptive_weight_only"` (`d_obs`-adaptive `lam` + uniform kNN weights)
   - `"full_gated_adaptive"` (`d_obs`-adaptive `lam` + expression edge gating)
   Note: Expression gating is treated as a testable hypothesis (it protects homotypic layer borders
   but could attenuate heterotypic niche contacts!).
4. **Scalable O(N * K_tot^2) Low-Rank Bipartite SVD Co-Association Consensus (`run_label_free_consensus`)**:
   Given target `n_domains` (label-free given K), avoids allocating the 40GB dense `N x N` co-association
   matrix `C = B @ B.T` by computing row degrees `d = B @ (B.T @ 1_N)` in `O(N * K_tot)` memory and
   taking the thin SVD of `D^{-1/2} @ B` (`shape N x K_tot`, where `K_tot = sum_m K_m << N`), yielding
   the exact top eigenvectors of `D^{-1/2} C D^{-1/2}` in linear memory `O(N * K_tot)`!
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV

from factor_demo.methods_zoo.base_pipeline import (
    MethodRunOutput,
    build_spatial_knn_graph,
    cluster_embedding,
    compute_effective_rank,
    preprocess_expression,
    standardize_branch,
)


class FactorAuditedChimera:
    """Modular case-study detector supporting 4-way component ablation (`variant`)."""

    VARIANTS = ("fixed_concat", "gated_only", "adaptive_weight_only", "full_gated_adaptive")

    def __init__(self) -> None:
        self.het_median_calib: float = 0.26
        self.is_calibrated: bool = False

    def calibrate_on_exploration(self, d_obs_expl_list: List[Dict[str, float]]) -> None:
        """Calibrate label-free heterogeneity midpoint strictly on Exploration cohort."""
        hets = [float(d["local_expr_heterogeneity"]) for d in d_obs_expl_list if "local_expr_heterogeneity" in d]
        if hets:
            self.het_median_calib = float(np.median(hets))
        self.is_calibrated = True

    def run(
        self,
        counts: np.ndarray,
        coords: np.ndarray,
        n_domains: int,
        d_obs_slice: Optional[Dict[str, float]] = None,
        variant: str = "full_gated_adaptive",
        preprocess_mode: str = "pca",
        seed: int = 0,
    ) -> MethodRunOutput:
        """Run specified ablation variant of FACTOR-Chimera (given target `n_domains`)."""
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown Chimera variant: {variant}. Expected one of {self.VARIANTS}")

        feats = preprocess_expression(counts, mode=preprocess_mode, n_hvgs=110, n_pcs=20, seed=seed)

        use_adaptive_lam = variant in ("adaptive_weight_only", "full_gated_adaptive")
        use_edge_gating = variant in ("gated_only", "full_gated_adaptive")

        loc_het = (
            float(d_obs_slice["local_expr_heterogeneity"])
            if (d_obs_slice and "local_expr_heterogeneity" in d_obs_slice)
            else self.het_median_calib
        )

        if use_adaptive_lam:
            lam_sp = 0.28 if loc_het > self.het_median_calib else 0.48
            k_nbr = 8 if loc_het > self.het_median_calib else 10
        else:
            lam_sp = 0.35
            k_nbr = 8

        adj_knn = build_spatial_knn_graph(coords, k_neighbors=k_nbr, include_self=False)

        if use_edge_gating:
            gate_temp = 2.5 if (use_adaptive_lam and loc_het > self.het_median_calib) else 1.8
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6
            f_unit = feats / norms
            cos_sim = f_unit @ f_unit.T
            att = np.exp(gate_temp * cos_sim) * (adj_knn > 0).astype(np.float32)
            agg_op = att / np.maximum(np.sum(att, axis=1, keepdims=True), 1e-6)
        else:
            gate_temp = 0.0
            agg_op = adj_knn

        # Standardize BOTH branches BEFORE weighting so `lam_sp` genuinely controls relative variance!
        z_self_std = standardize_branch(feats)
        z_nbr_std = standardize_branch(agg_op @ feats)

        emb = np.concatenate(
            [np.sqrt(1.0 - lam_sp) * z_self_std, np.sqrt(lam_sp) * z_nbr_std],
            axis=1,
        ).astype(np.float32)

        preds = cluster_embedding(
            emb,
            n_domains=n_domains,
            cluster_head="gmm",
            seed=seed,
            preserve_relative_weights=True,
        )
        return MethodRunOutput(
            pred_labels=preds,
            embedding=emb,
            param_count=int(emb.shape[1]),
            converged=True,
            final_loss=0.0,
            loss_drop_ratio=1.0,
            effective_rank=compute_effective_rank(emb),
            hyperparams_used={
                "variant": variant,
                "preprocess": preprocess_mode,
                "k_neighbors": k_nbr,
                "lambda_sp": lam_sp,
                "gate_temperature": gate_temp,
                "cluster_head": "gmm",
            },
        )


class FactorMethodRecommender:
    """Label-free `d_obs` meta-recommender with runtime cost accounting."""

    def __init__(self) -> None:
        self.method_ids: List[str] = []
        self.feature_keys: List[str] = []
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.models: List[RidgeCV] = []
        self.method_runtimes_sec: Dict[str, float] = {}

    def fit(
        self,
        d_obs_expl: List[Dict[str, float]],
        y_expl: np.ndarray,
        method_ids: List[str],
        method_runtimes_sec: Optional[Dict[str, float]] = None,
    ) -> None:
        self.method_ids = list(method_ids)
        self.feature_keys = sorted(d_obs_expl[0].keys())
        X = np.asarray([[float(d[k]) for k in self.feature_keys] for d in d_obs_expl], dtype=np.float64)
        self.mu = np.mean(X, axis=0, keepdims=True)
        self.sd = np.std(X, axis=0, keepdims=True) + 1e-6
        X_s = (X - self.mu) / self.sd

        self.models = []
        for j in range(len(method_ids)):
            reg = RidgeCV(alphas=np.logspace(-1, 2, 9))
            reg.fit(X_s, y_expl[:, j])
            self.models.append(reg)

        self.method_runtimes_sec = dict(method_runtimes_sec or {})

    def predict_scores(self, d_obs_slice: Dict[str, float]) -> Dict[str, float]:
        assert self.mu is not None and self.sd is not None
        x = np.asarray([[float(d_obs_slice[k]) for k in self.feature_keys]], dtype=np.float64)
        x_s = (x - self.mu) / self.sd
        return {
            mid: float(self.models[j].predict(x_s)[0])
            for j, mid in enumerate(self.method_ids)
        }

    def recommend(
        self,
        d_obs_slice: Dict[str, float],
        max_runtime_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        scores = self.predict_scores(d_obs_slice)
        eligible = [
            m for m in self.method_ids
            if max_runtime_sec is None or self.method_runtimes_sec.get(m, 0.0) <= max_runtime_sec
        ]
        if not eligible:
            eligible = list(self.method_ids)
        best_m = max(eligible, key=lambda m: scores[m])
        return {
            "recommended_method": best_m,
            "predicted_ari": round(scores[best_m], 4),
            "all_predicted_scores": {k: round(v, 4) for k, v in scores.items()},
        }


def run_label_free_consensus(
    predictions_by_method: Dict[str, np.ndarray],
    n_domains: int,
    method_weights: Optional[Dict[str, float]] = None,
    seed: int = 0,
) -> np.ndarray:
    """Scalable O(N * K_tot^2) Low-Rank Bipartite SVD Co-Association Consensus (given target `n_domains`).

    Avoids choosing an arbitrary reference clustering for label-ID alignment AND avoids
    allocating any dense `N x N` matrix (scales linearly in cell count `N` to >100,000 cells!):
    - Forms `B = [sqrt(w_1) H_1, ..., sqrt(w_M) H_M]` of shape `(N, K_tot)` where `K_tot = sum_m K_m`.
    - Implicit co-association is `C = B @ B.T`; exact row degrees are `d = B @ (B.T @ 1_N)`.
    - Left singular vectors `U` of `B_tilde = D^{-1/2} @ B` equal the exact top eigenvectors
      of `D^{-1/2} C D^{-1/2}`!
    """
    m_ids = sorted(predictions_by_method.keys())
    if not m_ids:
        raise ValueError("Empty predictions_by_method")

    n_units = len(predictions_by_method[m_ids[0]])
    if method_weights is None:
        weights = {m: 1.0 / len(m_ids) for m in m_ids}
    else:
        raw_w = np.array([max(0.05, float(method_weights.get(m, 1.0))) for m in m_ids], dtype=np.float64)
        exp_w = np.exp(3.0 * (raw_w - np.max(raw_w)))
        norm_w = exp_w / np.sum(exp_w)
        weights = {m: float(norm_w[idx]) for idx, m in enumerate(m_ids)}

    blocks = []
    for m in m_ids:
        y_m = np.asarray(predictions_by_method[m], dtype=np.int64)
        uniq, inv = np.unique(y_m, return_inverse=True)
        one_hot = np.zeros((n_units, len(uniq)), dtype=np.float32)
        one_hot[np.arange(n_units), inv] = np.sqrt(float(weights[m]))
        blocks.append(one_hot)

    # Shape (N, K_tot), e.g., (100_000, 40) -> ~16 MB instead of 40 GB!
    B = np.hstack(blocks)
    col_sums = np.sum(B, axis=0)            # shape (K_tot,)
    deg = B @ col_sums                      # exact row sums of B @ B.T in O(N * K_tot)!
    d_inv_sqrt = (1.0 / np.sqrt(np.maximum(deg, 1e-6))).astype(np.float32)
    B_tilde = B * d_inv_sqrt[:, None]

    # Thin SVD: O(N * K_tot^2) time, O(N * K_tot) memory
    U, _, _ = np.linalg.svd(B_tilde, full_matrices=False)
    k = min(n_domains, U.shape[1])
    top_vecs = U[:, :k]
    row_norms = np.linalg.norm(top_vecs, axis=1, keepdims=True) + 1e-6
    spec_emb = top_vecs / row_norms

    km = KMeans(n_clusters=n_domains, n_init=15, random_state=seed)
    return km.fit_predict(spec_emb).astype(np.int64)
