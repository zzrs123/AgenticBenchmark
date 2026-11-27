"""Parameterized, operationally defined data intervention operators with explicit label rules.

Addresses Multiple Versions of Treatment & Label Evolution Prerequisites:
1. `u1_spot_aggregation`: Coarsens single-cell coordinates into grid spots of side length `grid_size`.
   - Counts are summed/averaged within each spot; spot reference domain is majority underlying domain label.
2. `u2_boundary_infiltration`: Exchanges whole-cell expression profiles across spatial domain boundaries
   within a boundary band at controlled fraction `swap_rate` in [0, 0.6].
   - Keeps anatomical region labels `domain_labels` fixed to tissue geometry (testing region delineation
     under infiltrating cellular heterogeneity).
   - Strictly preserves every cell's multivariate gene vector (hence exact global marginal distribution
     AND exact global gene-gene correlation matrix!).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from factor_demo.data_lab.generator_and_loader import SpatialSlice


OPERATOR_SPECS: Dict[str, Dict[str, object]] = {
    "u2_boundary_infiltration": {
        "description": "Cross-boundary whole-cell expression exchange within anatomical border zones",
        "label_rule": "fixed_anatomical_region_geometry",
        "target_features": ["local_expr_heterogeneity", "global_morans_i"],
        "expected_covarying_features": ["boundary_sharpness_ref"],
        "invariant_features": ["sparsity_fraction", "mean_genes_per_unit", "spatial_density"],
        "default_doses": [0.0, 0.25, 0.50],
    },
    "u1_spot_aggregation": {
        "description": "Spatial grid binning from single-cell coordinates to coarser multi-cell spots",
        "label_rule": "majority_underlying_niche_vote",
        "target_features": ["spatial_density"],
        "expected_covarying_features": ["sparsity_fraction", "mean_genes_per_unit", "local_expr_heterogeneity"],
        "invariant_features": ["n_profiled_genes"],
        "default_doses": [1.0, 6.0, 11.0],
    },
}


def apply_boundary_infiltration(
    slc: SpatialSlice,
    swap_rate: float,
    seed: int = 0,
    k_boundary: int = 10,
) -> SpatialSlice:
    """Operator u2: Controlled cross-boundary cell expression exchange at dose `swap_rate`."""
    out = slc.copy()
    if swap_rate <= 1e-8:
        out.metadata["applied_operator"] = "u2_boundary_infiltration"
        out.metadata["applied_dose"] = 0.0
        return out

    rng = np.random.default_rng(seed)
    diff = out.coords[:, None, :] - out.coords[None, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    nn_idx = np.argsort(dist_mat, axis=1)[:, 1 : k_boundary + 1]

    # Candidate cells residing near any inter-domain border
    border_candidates: List[Tuple[int, int]] = []
    used = set()
    order = rng.permutation(len(out.domain_labels))
    for i in order:
        if i in used:
            continue
        yi = out.domain_labels[i]
        foreign_neighbors = [j for j in nn_idx[i] if out.domain_labels[j] != yi and j not in used]
        if foreign_neighbors:
            j = int(rng.choice(foreign_neighbors))
            border_candidates.append((int(i), j))
            used.add(int(i))
            used.add(j)

    n_pairs_to_swap = int(round(len(border_candidates) * min(1.0, max(0.0, swap_rate))))
    new_counts = out.counts.copy()
    for i, j in border_candidates[:n_pairs_to_swap]:
        new_counts[i], new_counts[j] = out.counts[j].copy(), out.counts[i].copy()

    out.counts = new_counts
    out.metadata["applied_operator"] = "u2_boundary_infiltration"
    out.metadata["applied_dose"] = float(swap_rate)
    out.metadata["swapped_pairs"] = n_pairs_to_swap
    return out


def apply_spot_aggregation(
    slc: SpatialSlice,
    grid_step: float,
    seed: int = 0,
) -> SpatialSlice:
    """Operator u1: Aggregate spatial units into square grid spots of side length `grid_step`."""
    out = slc.copy()
    if grid_step <= 1.05:
        out.metadata["applied_operator"] = "u1_spot_aggregation"
        out.metadata["applied_dose"] = float(grid_step)
        return out

    rng = np.random.default_rng(seed)
    # Slight random grid offset controlled by `seed` to account for discretization phase randomness
    offset = rng.uniform(0.0, grid_step, size=(1, 2)).astype(np.float32)
    bins = np.floor((out.coords + offset) / float(grid_step)).astype(np.int64)

    bucket_map: Dict[Tuple[int, int], List[int]] = {}
    for idx, (bx, by) in enumerate(bins):
        bucket_map.setdefault((int(bx), int(by)), []).append(idx)

    agg_counts = []
    agg_coords = []
    agg_labels = []
    for _, member_idx in sorted(bucket_map.items()):
        m_arr = np.array(member_idx, dtype=np.int64)
        agg_counts.append(np.sum(out.counts[m_arr], axis=0))
        agg_coords.append(np.mean(out.coords[m_arr], axis=0))
        vals, cnts = np.unique(out.domain_labels[m_arr], return_counts=True)
        agg_labels.append(int(vals[np.argmax(cnts)]))

    new_labels = np.array(agg_labels, dtype=np.int64)
    uniq_present = np.unique(new_labels)

    out.counts = np.asarray(agg_counts, dtype=np.float32)
    out.coords = np.asarray(agg_coords, dtype=np.float32)
    out.domain_labels = new_labels
    out.n_domains = max(out.n_domains, len(uniq_present))
    out.metadata["applied_operator"] = "u1_spot_aggregation"
    out.metadata["applied_dose"] = float(grid_step)
    return out


def apply_data_operator(
    slc: SpatialSlice,
    operator_name: str,
    dose: float,
    seed: int = 0,
) -> SpatialSlice:
    """Dispatch named operational data perturbation."""
    if operator_name == "u2_boundary_infiltration":
        return apply_boundary_infiltration(slc, swap_rate=dose, seed=seed)
    if operator_name == "u1_spot_aggregation":
        return apply_spot_aggregation(slc, grid_step=dose, seed=seed)
    raise ValueError(f"Unknown data operator: {operator_name}")
