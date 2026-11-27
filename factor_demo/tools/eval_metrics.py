"""Evaluation metrics for spatial domain identification (label-based & annotation-free orthogonal)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


def align_clusters_to_reference(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Align unsupervised cluster IDs to reference domain IDs via maximum-weight bipartite matching."""
    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)
    size = max(int(np.max(true_classes)) + 1, int(np.max(pred_classes)) + 1)

    conf = np.zeros((size, size), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1

    row_ind, col_ind = linear_sum_assignment(-conf)
    mapping = {int(col): int(row) for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(int(p), int(p)) for p in y_pred], dtype=np.int64)


def compute_pas_score(preds: np.ndarray, coords: np.ndarray, k: int = 10) -> float:
    """Percentage of Abnormal Spots (PAS): fraction of units differing from >50% of k spatial neighbors."""
    n = len(preds)
    k_eff = min(k, max(1, n - 1))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    nn = np.argsort(dist, axis=1)[:, 1 : k_eff + 1]
    disagree_frac = np.mean(preds[nn] != preds[:, None], axis=1)
    return float(np.mean(disagree_frac > 0.5))


def compute_boundary_f1(y_true: np.ndarray, y_pred: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    """Macro F1 evaluated specifically on anatomical border-zone units & minority thin domains."""
    aligned = align_clusters_to_reference(y_true, y_pred)
    n = len(y_true)
    k_eff = min(k, max(1, n - 1))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    nn = np.argsort(dist, axis=1)[:, 1 : k_eff + 1]
    border_mask = np.any(y_true[nn] != y_true[:, None], axis=1)

    if int(np.sum(border_mask)) < 10:
        return float(f1_score(y_true, aligned, average="macro", zero_division=0))
    return float(f1_score(y_true[border_mask], aligned[border_mask], average="macro", zero_division=0))


def compute_heldout_marker_enrichment(
    counts: np.ndarray,
    y_pred: np.ndarray,
    heldout_markers: Dict[int, List[int]],
) -> float:
    """Annotation-free biological validity metric: specificity of held-out domain marker genes across predicted clusters."""
    if not heldout_markers:
        return 0.0

    lib = np.sum(counts, axis=1, keepdims=True)
    log_expr = np.log1p((counts / np.maximum(lib, 1e-6)) * max(float(np.median(lib)), 1.0))
    pred_clusters = np.unique(y_pred)
    if len(pred_clusters) < 2:
        return 0.0

    scores = []
    for _, gene_idx in heldout_markers.items():
        valid_g = [g for g in gene_idx if g < log_expr.shape[1]]
        if not valid_g:
            continue
        sig = np.mean(log_expr[:, valid_g], axis=1)
        # Best cluster vs rest standardized contrast (independent of manual domain polygon labels!)
        best_contrast = 0.0
        global_std = float(np.std(sig)) + 1e-6
        for c in pred_clusters:
            mask = (y_pred == c)
            if 0 < np.sum(mask) < len(y_pred):
                contrast = (float(np.mean(sig[mask])) - float(np.mean(sig[~mask]))) / global_std
                best_contrast = max(best_contrast, contrast)
        scores.append(best_contrast)

    return float(np.mean(scores)) if scores else 0.0


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coords: np.ndarray,
    counts: np.ndarray,
    heldout_markers: Dict[int, List[int]],
) -> Dict[str, float]:
    """Compute comprehensive label-based and orthogonal evaluation metrics."""
    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    bf1 = compute_boundary_f1(y_true, y_pred, coords=coords)
    pas = compute_pas_score(y_pred, coords=coords)
    marker_enrich = compute_heldout_marker_enrichment(counts, y_pred, heldout_markers)
    return {
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "Boundary_F1": round(bf1, 4),
        "PAS": round(pas, 4),
        "Marker_Enrichment": round(marker_enrich, 4),
    }
