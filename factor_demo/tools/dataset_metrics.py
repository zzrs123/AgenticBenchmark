"""Deterministic spatial & transcriptional dataset meta-feature calculator and drift auditor.

Strictly separates:
- `d_obs`: Label-free observable features (safe for test-time method recommendation).
- `d_ref`: Ground-truth-dependent diagnostic features (only for retrospective benchmark attribution).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from factor_demo.data_lab.generator_and_loader import SpatialSlice


def _log_normalize(counts: np.ndarray) -> np.ndarray:
    lib = np.sum(counts, axis=1, keepdims=True)
    med = max(float(np.median(lib)), 1.0)
    return np.log1p((counts / np.maximum(lib, 1e-6)) * med)


def _compute_morans_i_top_genes(log_expr: np.ndarray, coords: np.ndarray, k: int = 8, n_top: int = 40) -> float:
    """Compute average spatial autocorrelation (Moran's I) across top variable genes."""
    n = log_expr.shape[0]
    k_eff = min(k, max(1, n - 1))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    nn = np.argsort(dist, axis=1)[:, 1 : k_eff + 1]

    W = np.zeros((n, n), dtype=np.float32)
    rows = np.repeat(np.arange(n), k_eff)
    W[rows, nn.reshape(-1)] = 1.0
    W_sum = float(np.sum(W))
    if W_sum <= 0:
        return 0.0

    variances = np.var(log_expr, axis=0)
    top_idx = np.argsort(variances)[-min(n_top, log_expr.shape[1]) :]
    X_sub = log_expr[:, top_idx]
    X_centered = X_sub - np.mean(X_sub, axis=0, keepdims=True)

    num = np.sum(X_centered * (W @ X_centered), axis=0)
    den = np.sum(X_centered ** 2, axis=0) + 1e-8
    morans = (n / W_sum) * (num / den)
    return float(np.mean(morans))


def compute_dataset_features(slc: SpatialSlice) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute (`d_obs`, `d_ref`) deterministically using strictly `get_model_input_counts()`.

    Excludes `heldout_markers` from ALL meta-features (`d_obs` & `d_ref`) so neither model
    training, meta-feature screening, recommender fitting, nor Chimera calibration ever touches
    held-out biological validation markers!
    """
    counts = slc.get_model_input_counts()
    coords = slc.coords
    labels = slc.domain_labels
    n_units, n_genes = counts.shape

    log_expr = _log_normalize(counts)

    # Spatial kNN geometry
    k_eff = min(8, max(1, n_units - 1))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    sorted_idx = np.argsort(dist, axis=1)
    nn_idx = sorted_idx[:, 1 : k_eff + 1]
    nn_dists = np.take_along_axis(dist, nn_idx, axis=1)

    median_nn_dist = max(float(np.median(nn_dists)), 1e-4)
    spatial_density = float(1.0 / median_nn_dist)
    sparsity_fraction = float(np.mean(counts <= 1e-8))
    mean_genes_per_unit = float(np.mean(np.sum(counts > 1e-8, axis=1)))
    global_morans_i = _compute_morans_i_top_genes(log_expr, coords, k=k_eff, n_top=40)

    # Label-free local transcriptional heterogeneity (1 - cosine similarity to spatial neighbors)
    norms = np.linalg.norm(log_expr, axis=1, keepdims=True) + 1e-6
    unit_vec = log_expr / norms
    nbr_mean_vec = np.mean(unit_vec[nn_idx], axis=1)
    local_cos_sim = np.sum(unit_vec * nbr_mean_vec, axis=1)
    local_expr_heterogeneity = float(np.mean(1.0 - local_cos_sim))

    d_obs = {
        "spatial_density": round(spatial_density, 4),
        "sparsity_fraction": round(sparsity_fraction, 4),
        "mean_genes_per_unit": round(mean_genes_per_unit, 2),
        "n_profiled_genes": float(n_genes),
        "global_morans_i": round(global_morans_i, 4),
        "local_expr_heterogeneity": round(local_expr_heterogeneity, 4),
    }

    # Reference-dependent diagnostics (never exposed to test-time recommender)
    uniq_doms, dom_counts = np.unique(labels, return_counts=True)
    min_domain_fraction_ref = float(np.min(dom_counts) / n_units)

    # Fraction of spatial neighbor edges crossing distinct ground-truth domains
    cross_edges = (labels[:, None] != labels[nn_idx]).astype(np.float32)
    boundary_edge_ratio_ref = float(np.mean(cross_edges))

    # Standardized centroid distance across ground-truth domains
    global_std = np.std(log_expr, axis=0) + 1e-5
    centroids = np.array([np.mean(log_expr[labels == d] / global_std, axis=0) for d in uniq_doms])
    if len(centroids) > 1:
        c_diff = centroids[:, None, :] - centroids[None, :, :]
        c_dist = np.sqrt(np.mean(c_diff ** 2, axis=2))
        triu = c_dist[np.triu_indices(len(centroids), k=1)]
        inter_domain_sep_ref = float(np.mean(triu))
    else:
        inter_domain_sep_ref = 0.0

    d_ref = {
        "min_domain_fraction_ref": round(min_domain_fraction_ref, 4),
        "boundary_edge_ratio_ref": round(boundary_edge_ratio_ref, 4),
        "inter_domain_sep_ref": round(inter_domain_sep_ref, 4),
    }
    return d_obs, d_ref


def compute_gene_corr_frobenius_drift(orig_counts: np.ndarray, pert_counts: np.ndarray, n_top: int = 50) -> float:
    """Measure Frobenius drift of the gene-gene correlation matrix on top variable genes."""
    log_orig = _log_normalize(orig_counts)
    log_pert = _log_normalize(pert_counts)
    top_idx = np.argsort(np.var(log_orig, axis=0))[-min(n_top, log_orig.shape[1]) :]

    c1 = np.corrcoef(log_orig[:, top_idx], rowvar=False)
    c2 = np.corrcoef(log_pert[:, top_idx], rowvar=False)
    c1 = np.nan_to_num(c1, nan=0.0)
    c2 = np.nan_to_num(c2, nan=0.0)
    k = c1.shape[0]
    return float(np.linalg.norm(c1 - c2, ord="fro") / max(1, k))


def audit_perturbation_drift(
    orig_slice: SpatialSlice,
    pert_slice: SpatialSlice,
    feature_scales: Dict[str, float],
    target_features: List[str],
    invariant_features: List[str],
) -> Dict[str, object]:
    """Compute Intervention Validity & Off-Target Drift Diagnostics (never over-claimed as causal proof)."""
    d_obs_0, d_ref_0 = compute_dataset_features(orig_slice)
    d_obs_1, d_ref_1 = compute_dataset_features(pert_slice)
    all_0 = {**d_obs_0, **d_ref_0}
    all_1 = {**d_obs_1, **d_ref_1}

    target_shifts = {
        f: abs(all_1[f] - all_0[f]) / max(feature_scales.get(f, 1.0), 1e-4)
        for f in target_features
        if f in all_0
    }
    off_drifts = {
        f: abs(all_1[f] - all_0[f]) / max(feature_scales.get(f, 1.0), 1e-4)
        for f in invariant_features
        if f in all_0
    }
    corr_drift = compute_gene_corr_frobenius_drift(orig_slice.counts, pert_slice.counts)

    max_target_z = max(target_shifts.values()) if target_shifts else 0.0
    max_off_z = max(off_drifts.values()) if off_drifts else 0.0

    return {
        "target_shifts_z": target_shifts,
        "max_target_shift_z": round(float(max_target_z), 4),
        "off_target_drifts_z": {k: round(float(v), 4) for k, v in off_drifts.items()},
        "max_off_target_drift_z": round(float(max_off_z), 4),
        "gene_corr_frobenius_drift": round(float(corr_drift), 5),
    }
