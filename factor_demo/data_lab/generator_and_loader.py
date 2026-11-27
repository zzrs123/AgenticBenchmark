"""Spatial transcriptomics slice container, .h5ad loader, and realistic cohort generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SpatialSlice:
    """In-memory representation of a spatial transcriptomics tissue section."""

    slice_id: str
    split: str  # "exploration" or "confirmation"
    donor_id: str
    platform_tag: str
    tissue_tag: str
    counts: np.ndarray          # shape (n_units, n_genes), non-negative float/int
    coords: np.ndarray          # shape (n_units, 2)
    domain_labels: np.ndarray   # shape (n_units,), integer 0..K-1
    n_domains: int
    gene_names: List[str]
    # Mapping from domain_id -> list of held-out marker gene indices (excluded from training if desired or audited orthogonally)
    heldout_markers: Dict[int, List[int]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_model_input_counts(self) -> np.ndarray:
        """Return expression matrix with all `heldout_markers` columns strictly excluded from training!"""
        if not self.heldout_markers:
            return self.counts
        withheld = {idx for indices in self.heldout_markers.values() for idx in indices}
        keep = [j for j in range(self.counts.shape[1]) if j not in withheld]
        return self.counts[:, keep]

    def copy(self) -> "SpatialSlice":
        return SpatialSlice(
            slice_id=self.slice_id,
            split=self.split,
            donor_id=self.donor_id,
            platform_tag=self.platform_tag,
            tissue_tag=self.tissue_tag,
            counts=self.counts.copy(),
            coords=self.coords.copy(),
            domain_labels=self.domain_labels.copy(),
            n_domains=self.n_domains,
            gene_names=list(self.gene_names),
            heldout_markers={k: list(v) for k, v in self.heldout_markers.items()},
            metadata=dict(self.metadata),
        )


def load_h5ad_slice(
    h5ad_path: str | Path,
    slice_id: Optional[str] = None,
    split: str = "exploration",
    label_key_candidates: Tuple[str, ...] = ("domain", "ground_truth", "layer_guess", "region", "cluster"),
    spatial_key: str = "spatial",
) -> SpatialSlice:
    """Load a real AnnData (.h5ad) slice (e.g., Visium DLPFC, MERFISH, Xenium)."""
    import anndata as ad
    import scipy.sparse as sp

    path = Path(h5ad_path)
    adata = ad.read_h5ad(path)
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    X = np.asarray(X, dtype=np.float32)

    if spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key][:, :2], dtype=np.float32)
    elif {"x", "y"}.issubset(adata.obs.columns):
        coords = adata.obs[["x", "y"]].to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Could not find spatial coordinates in {path}")

    label_col = None
    for cand in label_key_candidates:
        if cand in adata.obs.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError(f"Could not find ground-truth domain column among {label_key_candidates} in {path}")

    raw_labels = adata.obs[label_col].astype(str).to_numpy()
    valid_mask = (raw_labels != "nan") & (raw_labels != "NA") & (raw_labels != "None")
    X = X[valid_mask]
    coords = coords[valid_mask]
    raw_labels = raw_labels[valid_mask]

    uniq = sorted(set(raw_labels))
    label_map = {name: idx for idx, name in enumerate(uniq)}
    y = np.array([label_map[name] for name in raw_labels], dtype=np.int64)

    gene_names = [str(g) for g in adata.var_names]
    return SpatialSlice(
        slice_id=slice_id or path.stem,
        split=split,
        donor_id=str(adata.obs["donor"].iloc[0]) if "donor" in adata.obs.columns else path.stem,
        platform_tag=str(adata.uns.get("platform", "h5ad_real")),
        tissue_tag=str(adata.uns.get("tissue", "real_tissue")),
        counts=X,
        coords=coords,
        domain_labels=y,
        n_domains=len(uniq),
        gene_names=gene_names,
        heldout_markers={},
        metadata={"source_path": str(path), "label_column": label_col},
    )


def _generate_single_slice(
    rng: np.random.Generator,
    slice_id: str,
    split: str,
    donor_id: str,
    platform_tag: str,
    tissue_tag: str,
    layout: str,
    n_units: int = 900,
    n_genes: int = 320,
    n_domains: int = 4,
    infiltration_rate: float = 0.05,
    thin_middle_width: float = 0.22,
    dropout_prob: float = 0.35,
    transcriptional_sep: float = 1.15,
) -> SpatialSlice:
    """Generate a realistic spatial slice with co-expressed gene modules and held-out markers."""
    coords = rng.uniform(0.0, 100.0, size=(n_units, 2)).astype(np.float32)
    x_norm = coords[:, 0] / 100.0
    y_norm = coords[:, 1] / 100.0

    # 1. Assign anatomical domain geometry
    if layout == "laminar_thick":
        edges = np.linspace(0.0, 1.0, n_domains + 1)
        domain_labels = np.digitize(x_norm, edges[1:-1]).astype(np.int64)
    elif layout == "laminar_thin_sandwich":
        # Domain 0 (left), Domain 1 (thin middle layer!), Domain 2 (right), Domain 3 (outer rim)
        c_mid = 0.45
        half_w = thin_middle_width / 2.0
        domain_labels = np.zeros(n_units, dtype=np.int64)
        domain_labels[(x_norm >= c_mid - half_w) & (x_norm < c_mid + half_w)] = 1
        domain_labels[(x_norm >= c_mid + half_w) & (x_norm < 0.78)] = 2
        domain_labels[x_norm >= 0.78] = min(3, n_domains - 1)
    elif layout == "concentric_tme":
        dist = np.sqrt((x_norm - 0.5) ** 2 + (y_norm - 0.5) ** 2)
        domain_labels = np.zeros(n_units, dtype=np.int64)
        domain_labels[(dist >= 0.16) & (dist < 0.28)] = 1  # Thin immune ring
        domain_labels[(dist >= 0.28) & (dist < 0.42)] = 2
        domain_labels[dist >= 0.42] = min(3, n_domains - 1)
    else:  # "pockets_heterogeneous"
        centers = np.array([[0.25, 0.28], [0.73, 0.27], [0.50, 0.72], [0.22, 0.75]])[:n_domains]
        dists = np.linalg.norm(coords[:, None, :] / 100.0 - centers[None, :, :], axis=2)
        domain_labels = np.argmin(dists, axis=1).astype(np.int64)

    # 2. Construct cell transcriptional identities (allowing boundary infiltration / mixing)
    cell_types = domain_labels.copy()
    # Identify units near spatial boundaries using k=8 nearest neighbors
    diff_coord = coords[:, None, :] - coords[None, :, :]
    dist_mat = np.sqrt(np.sum(diff_coord ** 2, axis=2))
    nn_idx = np.argsort(dist_mat, axis=1)[:, 1:9]
    is_boundary = np.any(domain_labels[nn_idx] != domain_labels[:, None], axis=1)

    for idx in np.where(is_boundary)[0]:
        if rng.random() < min(0.85, infiltration_rate * 2.2):
            neighbor_labels = domain_labels[nn_idx[idx]]
            foreign = neighbor_labels[neighbor_labels != domain_labels[idx]]
            if len(foreign) > 0:
                cell_types[idx] = rng.choice(foreign)

    # Global background infiltrating cells (e.g., scattered immune/stromal cells)
    bg_mask = rng.random(n_units) < (infiltration_rate * 0.5)
    cell_types[bg_mask] = rng.integers(0, n_domains, size=int(np.sum(bg_mask)))

    # 3. Simulate realistic gene expression with shared latent factors (preserving gene-gene covariance)
    n_factors = 6
    factor_loadings = rng.normal(0.0, 0.55, size=(n_factors, n_genes))
    domain_factor_centers = rng.normal(0.0, transcriptional_sep, size=(n_domains, n_factors))
    # Make Domain 0 and Domain 1 transcriptionally closer in thin-sandwich slices to test oversmoothing collapse
    if layout in ("laminar_thin_sandwich", "concentric_tme") and n_domains >= 3:
        domain_factor_centers[1] = 0.62 * domain_factor_centers[0] + 0.38 * domain_factor_centers[1]

    cell_factors = domain_factor_centers[cell_types] + rng.normal(0.0, 0.65, size=(n_units, n_factors))
    log_rates = 0.4 + cell_factors @ factor_loadings

    # Inject clean held-out domain marker genes (last 5 * n_domains genes) for orthogonal biological evaluation
    heldout_markers: Dict[int, List[int]] = {}
    marker_start = n_genes - 5 * n_domains
    for d in range(n_domains):
        m_indices = list(range(marker_start + d * 5, marker_start + (d + 1) * 5))
        heldout_markers[d] = m_indices
        in_dom = (cell_types == d)
        log_rates[:, m_indices] = -0.3
        log_rates[in_dom][:, m_indices] = 1.45

    # Library size variation + Poisson/Gamma (Negative Binomial) counts + technical dropout
    lib_scale = rng.lognormal(mean=0.0, sigma=0.28, size=(n_units, 1))
    mu = np.clip(np.exp(log_rates) * lib_scale, 0.02, 60.0)
    gamma_rates = rng.gamma(shape=2.5, scale=mu / 2.5)
    counts = rng.poisson(gamma_rates).astype(np.float32)

    if dropout_prob > 0.0:
        drop_mask = rng.random(counts.shape) < dropout_prob
        counts[drop_mask] = 0.0

    gene_names = [f"Gene_{g:03d}" for g in range(marker_start)] + [
        f"Marker_D{d}_{k}" for d in range(n_domains) for k in range(5)
    ]

    return SpatialSlice(
        slice_id=slice_id,
        split=split,
        donor_id=donor_id,
        platform_tag=platform_tag,
        tissue_tag=tissue_tag,
        counts=counts,
        coords=coords,
        domain_labels=domain_labels,
        n_domains=n_domains,
        gene_names=gene_names,
        heldout_markers=heldout_markers,
        metadata={
            "layout": layout,
            "sim_infiltration_rate": infiltration_rate,
            "sim_dropout_prob": dropout_prob,
            "thin_middle_width": thin_middle_width,
        },
    )


def build_demo_cohort(seed: int = 42) -> List[SpatialSlice]:
    """Build 8 Exploration slices + 4 independent Held-out Confirmation slices across distinct regimes."""
    rng = np.random.default_rng(seed)
    specs = [
        # --- Exploration Split (8 slices across 4 donors/platforms) ---
        ("S01_Cortex_Thick_A", "exploration", "Donor_E1", "MERFISH_sc", "Brain_Cortex", "laminar_thick", 850, 300, 0.04, 0.25, 0.25, 1.25),
        ("S02_Cortex_Thick_B", "exploration", "Donor_E1", "MERFISH_sc", "Brain_Cortex", "laminar_thick", 850, 300, 0.06, 0.25, 0.30, 1.15),
        ("S03_Visium_Laminar", "exploration", "Donor_E2", "Visium_spot", "DLPFC_Like", "laminar_thick", 750, 420, 0.08, 0.25, 0.55, 1.20),
        ("S04_SlideSeq_Sparse", "exploration", "Donor_E2", "SlideSeq_10um", "Cerebellum", "laminar_thick", 850, 380, 0.10, 0.22, 0.78, 1.10),
        ("S05_ThinLayer_Hypo", "exploration", "Donor_E3", "Xenium_sc", "Hypothalamus", "laminar_thin_sandwich", 900, 280, 0.18, 0.10, 0.22, 1.05),
        ("S06_TME_Ring_Breast", "exploration", "Donor_E3", "Xenium_sc", "Breast_TME", "concentric_tme", 900, 280, 0.26, 0.12, 0.25, 1.05),
        ("S07_TME_Infil_CRC", "exploration", "Donor_E4", "CosMx_sc", "CRC_Invasive", "pockets_heterogeneous", 900, 320, 0.32, 0.14, 0.35, 1.10),
        ("S08_TME_Sparse_Lung", "exploration", "Donor_E4", "SlideSeq_10um", "Lung_TME", "concentric_tme", 850, 360, 0.28, 0.12, 0.72, 1.05),
        # --- Held-Out Confirmation Split (4 independent donors for pre-registered factorial tests!) ---
        ("C01_Holdout_Cortex", "confirmation", "Donor_C1", "MERFISH_sc", "Brain_Cortex", "laminar_thick", 900, 300, 0.05, 0.24, 0.25, 1.20),
        ("C02_Holdout_ThinLayer", "confirmation", "Donor_C2", "Xenium_sc", "Hypothalamus", "laminar_thin_sandwich", 920, 300, 0.12, 0.11, 0.24, 1.08),
        ("C03_Holdout_TME_Ring", "confirmation", "Donor_C3", "Xenium_sc", "Breast_TME", "concentric_tme", 920, 300, 0.15, 0.12, 0.26, 1.08),
        ("C04_Holdout_CRC_Pockets", "confirmation", "Donor_C4", "CosMx_sc", "CRC_Invasive", "pockets_heterogeneous", 920, 300, 0.16, 0.14, 0.30, 1.10),
        # --- External Downstream Test Split (3 untouched donors reserved ONLY for downstream tools!) ---
        ("T01_External_Cortex", "external_test", "Donor_T1", "STARmap_sc", "Visual_Cortex", "laminar_thick", 900, 300, 0.06, 0.23, 0.26, 1.18),
        ("T02_External_ThinSandwich", "external_test", "Donor_T2", "Xenium_sc", "Retina_Layers", "laminar_thin_sandwich", 920, 300, 0.14, 0.11, 0.25, 1.06),
        ("T03_External_TME_Invasive", "external_test", "Donor_T3", "CosMx_sc", "Melanoma_TME", "concentric_tme", 920, 300, 0.22, 0.12, 0.28, 1.08),
    ]

    slices: List[SpatialSlice] = []
    for sid, split, donor, plat, tissue, layout, n_u, n_g, infil, width, drop, sep in specs:
        slices.append(
            _generate_single_slice(
                rng=rng,
                slice_id=sid,
                split=split,
                donor_id=donor,
                platform_tag=plat,
                tissue_tag=tissue,
                layout=layout,
                n_units=n_u,
                n_genes=n_g,
                n_domains=4,
                infiltration_rate=infil,
                thin_middle_width=width,
                dropout_prob=drop,
                transcriptional_sep=sep,
            )
        )
    return slices
