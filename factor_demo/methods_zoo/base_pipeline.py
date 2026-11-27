"""Shared modular building blocks & 4-arm counterfactual graph hooks for spatial domain methods.

Supports explicit separation of:
- `orig`: Unmodified default spatial graph computation.
- `refac`: Hooked pipeline preserving exact spatial graph (code-equivalence audit arm).
- `self`: Identity adjacency matrix `I` (tests cross-unit aggregation vs. cell-autonomous transform).
- `null`: Node-permuted isomorphic graph `P A_spat P^T` (preserves exact spectrum & degree sequence
          while breaking physical coordinate alignment).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


@dataclass
class MethodRunOutput:
    """Standardized method output with optimization & representation diagnostics."""

    pred_labels: np.ndarray
    embedding: np.ndarray
    param_count: int
    converged: bool
    final_loss: float
    loss_drop_ratio: float
    effective_rank: float
    hyperparams_used: Dict[str, object]


def preprocess_expression(
    counts: np.ndarray,
    mode: str = "pca",
    n_hvgs: int = 120,
    n_pcs: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Normalize counts (library-size + log1p) and extract either PCA or HVG features."""
    lib = np.sum(counts, axis=1, keepdims=True)
    median_lib = max(float(np.median(lib)), 1.0)
    normed = np.log1p((counts / np.maximum(lib, 1e-6)) * median_lib)

    # Variance-stabilized z-score per gene
    mean = np.mean(normed, axis=0, keepdims=True)
    std = np.std(normed, axis=0, keepdims=True) + 1e-6
    z = np.clip((normed - mean) / std, -8.0, 8.0)

    if mode == "hvg":
        # Dispersion ranking on log1p normalized counts
        gene_vars = np.var(normed, axis=0)
        k = min(n_hvgs, counts.shape[1])
        top_idx = np.argsort(gene_vars)[-k:]
        return z[:, top_idx].astype(np.float32)

    if mode == "pca":
        k = min(n_pcs, z.shape[0] - 1, z.shape[1])
        pca = PCA(n_components=max(2, k), random_state=seed)
        return pca.fit_transform(z).astype(np.float32)

    raise ValueError(f"Unsupported preprocess mode: {mode}")


def build_spatial_knn_graph(
    coords: np.ndarray,
    k_neighbors: int = 8,
    include_self: bool = True,
) -> np.ndarray:
    """Construct symmetric row-normalized spatial k-NN adjacency matrix."""
    n = coords.shape[0]
    k = min(k_neighbors, max(1, n - 1))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    nn = np.argsort(dist, axis=1)[:, 1 : k + 1]

    adj = np.zeros((n, n), dtype=np.float32)
    rows = np.repeat(np.arange(n), k)
    adj[rows, nn.reshape(-1)] = 1.0
    adj = np.maximum(adj, adj.T)
    if include_self:
        np.fill_diagonal(adj, 1.0)

    deg = np.sum(adj, axis=1, keepdims=True)
    return adj / np.maximum(deg, 1e-6)


def apply_graph_arm_hook(
    adj_spatial: np.ndarray,
    arm: str = "orig",
    null_seed: int = 0,
) -> np.ndarray:
    """Apply counterfactual method-side graph operator arm."""
    n = adj_spatial.shape[0]
    if arm in ("orig", "refac"):
        return adj_spatial.copy()
    if arm == "self":
        return np.eye(n, dtype=np.float32)
    if arm == "null":
        rng = np.random.default_rng(null_seed)
        perm = rng.permutation(n)
        # Isomorphic permutation P A P^T: exact same eigenvalues and degree distribution
        adj_perm = adj_spatial[perm][:, perm].copy()
        return adj_perm
    raise ValueError(f"Unknown method arm: {arm}")


def standardize_branch(branch: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance standardize a feature branch BEFORE applying concatenation weights."""
    mu = np.mean(branch, axis=0, keepdims=True)
    sd = np.std(branch, axis=0, keepdims=True) + 1e-6
    return ((branch - mu) / sd).astype(np.float32)


def cluster_embedding(
    embedding: np.ndarray,
    n_domains: int,
    cluster_head: str = "gmm",
    seed: int = 0,
    preserve_relative_weights: bool = True,
) -> np.ndarray:
    """Cluster low-dimensional representation into `n_domains` clusters.

    CRITICAL: When `preserve_relative_weights=True` (default), centers columns and scales the
    entire matrix by a SINGLE global scalar RMS norm rather than per-column standard deviation,
    so deliberate branch weights (e.g. `[sqrt(1-lam)*Z_self, sqrt(lam)*Z_nbr]`) are strictly preserved!
    """
    centered = embedding - np.mean(embedding, axis=0, keepdims=True)
    if preserve_relative_weights:
        global_scale = float(np.sqrt(np.mean(centered ** 2))) + 1e-5
        emb = centered / global_scale
    else:
        emb = centered / (np.std(centered, axis=0, keepdims=True) + 1e-5)
    if cluster_head == "gmm":
        gmm = GaussianMixture(
            n_components=n_domains,
            covariance_type="diag",
            reg_covar=1e-3,
            n_init=2,
            random_state=seed,
        )
        return gmm.fit_predict(emb).astype(np.int64)
    if cluster_head == "kmeans":
        km = KMeans(n_clusters=n_domains, n_init=10, random_state=seed)
        return km.fit_predict(emb).astype(np.int64)
    raise ValueError(f"Unknown cluster head: {cluster_head}")


def compute_effective_rank(embedding: np.ndarray) -> float:
    """Compute Shannon entropy-based effective rank of the learned representation."""
    centered = embedding - np.mean(embedding, axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    s_pos = s[s > 1e-8]
    if len(s_pos) == 0:
        return 1.0
    p = s_pos / np.sum(s_pos)
    entropy = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.exp(entropy))


def unsupervised_silhouette_proxy(embedding: np.ndarray, labels: np.ndarray) -> float:
    """Label-free cluster compactness score used during matched-budget hyperparameter retuning."""
    from sklearn.metrics import calinski_harabasz_score

    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= len(labels):
        return -1e6
    return float(calinski_harabasz_score(embedding, labels))
