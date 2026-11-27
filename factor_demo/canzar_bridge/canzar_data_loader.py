"""Data & Simulator Bridge for Descoeudres et al. (March 2026) repositories.

Provides three capabilities:
1. Load any real tissue slice formatted in Canzar Lab's 3-CSV standard:
   `resources/<dataset>/<sample>_UMImatrix.csv`, `_coords.csv`, `_labels.csv`
   (works out-of-the-box on `resources/osmFISH_Codeluppi/correctedsample_*`, and includes
   a converter from `.h5ad` to Canzar 3-CSV format for Visium DLPFC / MERFISH / Xenium).
2. Drive `external/spatial-domain-simulator` (`subset_anndata.h5ad.gz` + `create_tissues.py`
   + `assign_counts.py` + `define_experiments.py`) to generate official semi-synthetic slices
   with controlled perturbations (resolution, gene panel, sparsity, Type I/II noise, thickness).
3. Parse `results/scores.csv` + `config/technology_meta.csv` in pure Python (zero numpy needed
   for metadata/scores parsing) to extract label-free landmarking features (`d_obs`) and
   ground-truth structural diagnostics (`d_ref`) across all 94 real tissue sections!
"""

from __future__ import annotations

import csv
import gzip
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FRAMEWORK_DIR = PROJECT_ROOT / "external" / "spatial-domain-analysis-framework"
DEFAULT_SIMULATOR_DIR = PROJECT_ROOT / "external" / "spatial-domain-simulator"

REAL_DATASETS_2026 = [
    "osmFISH_Codeluppi",
    "MERFISH_Moffitt",
    "MERFISH_Zhang",
    "STARmap_Wang",
    "Slide-seq_Langlieb",
    "Slide-seqV2_Lan",
    "StereoSeq_Chen",
    "10xVisium_Fu",
    "10xVisium_Maynard",
    "ST_Stahl",
]

# Dataset-level metadata from Table 1 & technology_meta.csv of Descoeudres et al. (2026)
DATASET_TECH_SPECS: Dict[str, Dict[str, Any]] = {
    "osmFISH_Codeluppi": {"tech": "osmFISH", "modality": "imaging", "is_single_cell": 1.0, "approx_genes": 33.0, "approx_spots": 4839.0, "organ": "Mouse_Somatosensory_Cortex"},
    "MERFISH_Moffitt": {"tech": "MERFISH", "modality": "imaging", "is_single_cell": 1.0, "approx_genes": 155.0, "approx_spots": 5663.0, "organ": "Mouse_Hypothalamus"},
    "MERFISH_Zhang": {"tech": "MERFISH", "modality": "imaging", "is_single_cell": 1.0, "approx_genes": 254.0, "approx_spots": 4890.0, "organ": "Mouse_Motor_Cortex"},
    "STARmap_Wang": {"tech": "STARmap", "modality": "in_situ_seq", "is_single_cell": 1.0, "approx_genes": 1020.0, "approx_spots": 1207.0, "organ": "Mouse_Visual_Cortex"},
    "Slide-seq_Langlieb": {"tech": "Slide-seq", "modality": "seq_highres", "is_single_cell": 0.5, "approx_genes": 27719.0, "approx_spots": 3447.0, "organ": "Mouse_Whole_Brain"},
    "Slide-seqV2_Lan": {"tech": "Slide-seqV2", "modality": "seq_highres", "is_single_cell": 0.5, "approx_genes": 22000.0, "approx_spots": 4000.0, "organ": "Mouse_Brain_SlideSeqV2"},
    "StereoSeq_Chen": {"tech": "StereoSeq", "modality": "seq_highres", "is_single_cell": 0.5, "approx_genes": 24000.0, "approx_spots": 5000.0, "organ": "Mouse_Embryo_Brain"},
    "10xVisium_Fu": {"tech": "Visium", "modality": "seq_spot50um", "is_single_cell": 0.0, "approx_genes": 36601.0, "approx_spots": 3798.0, "organ": "Human_Breast_Cancer"},
    "10xVisium_Maynard": {"tech": "Visium", "modality": "seq_spot50um", "is_single_cell": 0.0, "approx_genes": 33538.0, "approx_spots": 3944.0, "organ": "Human_DLPFC"},
    "ST_Stahl": {"tech": "ST", "modality": "seq_spot100um", "is_single_cell": 0.0, "approx_genes": 15733.0, "approx_spots": 260.0, "organ": "Mouse_Olfactory_Bulb"},
}


def inspect_raw_3csv_slice_pure_python(
    framework_dir: Path = DEFAULT_FRAMEWORK_DIR,
    dataset_name: str = "osmFISH_Codeluppi",
    sample_name: str = "correctedsample",
) -> Dict[str, Any]:
    """Compute exact count-matrix & spatial statistics on a Canzar 3-CSV slice using pure Python."""
    base = framework_dir / "resources" / dataset_name / sample_name
    umi_file = Path(f"{base}_UMImatrix.csv")
    coord_file = Path(f"{base}_coords.csv")
    label_file = Path(f"{base}_labels.csv")

    coords: List[Tuple[float, float]] = []
    with coord_file.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                coords.append((float(row[0]), float(row[1])))

    labels: List[str] = []
    with label_file.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                labels.append(row[0].strip())

    n_cells = 0
    n_genes = 0
    zero_entries = 0
    total_entries = 0
    nonzero_per_cell: List[int] = []
    lib_sizes: List[float] = []

    with umi_file.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            vals = [float(x) for x in row]
            if n_genes == 0:
                n_genes = len(vals)
            n_cells += 1
            nz = sum(1 for v in vals if v > 0)
            zeros = len(vals) - nz
            zero_entries += zeros
            total_entries += len(vals)
            nonzero_per_cell.append(nz)
            lib_sizes.append(sum(vals))

    dom_counts: Dict[str, int] = defaultdict(int)
    for lb in labels:
        dom_counts[lb] += 1

    min_dom_frac = min(dom_counts.values()) / max(1, n_cells)
    sparsity = zero_entries / max(1, total_entries)
    mean_genes_detected = sum(nonzero_per_cell) / max(1, n_cells)
    mean_umi = sum(lib_sizes) / max(1, n_cells)

    return {
        "dataset": dataset_name,
        "sample": sample_name,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_domains": len(dom_counts),
        "sparsity_fraction": round(sparsity, 4),
        "mean_genes_per_cell": round(mean_genes_detected, 2),
        "mean_total_counts_per_cell": round(mean_umi, 2),
        "min_domain_fraction": round(min_dom_frac, 4),
        "domain_distribution": dict(sorted(dom_counts.items(), key=lambda kv: kv[1], reverse=True)),
    }


def load_canzar_real_benchmark_records(
    framework_dir: Path = DEFAULT_FRAMEWORK_DIR,
    exclude_st_stahl: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Parse all real-tissue benchmark runs + extract `d_obs` and `d_ref` per sample from `scores.csv`.

    How `d_obs` (strictly label-free!) vs `d_ref` (reference-dependent!) are extracted:
    - `d_obs` (usable on unannotated slices):
      1. `is_single_cell`: platform physical resolution indicator (1.0=sc, 0.5=10um, 0.0=50-100um)
      2. `log10_panel_genes`: log10 number of profiled genes
      3. `scanpy_PAS_spatial`: spatial fragmentation (PAS) of purely non-spatial `scanpy` expression clusters!
         (A 100% label-free landmarking probe measuring how spatially smooth expression clusters are!)
      4. `scanpy_SI_expr`: transcriptional cluster silhouette on UMAP (`SI_X`) of `scanpy` clusters
      5. `scanpy_CHAOS_spatial`: spatial dispersion (`CHAOS`) of non-spatial `scanpy` clusters
    - `d_ref` (ground-truth diagnostic features, never leaked to test-time recommender):
      1. `gt_PAS_spatial`: ground-truth domain boundary irregularity (`PAS` of `gt`)
      2. `gt_RAS_spatial`: ground-truth broader neighborhood mixing (`RAS` of `gt`, k=40)
      3. `gt_SI_expr`: transcriptional separability of true anatomical domains (`SI_X` of `gt`)
      4. `gt_CHAOS_spatial`: spatial shape complexity (`CHAOS` of `gt`)
    """
    scores_path = framework_dir / "results" / "scores.csv"
    target_ds = [d for d in REAL_DATASETS_2026 if not (exclude_st_stahl and d == "ST_Stahl")]

    by_sample_method: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = defaultdict(dict)
    raw_rows: List[Dict[str, Any]] = []

    def _safe_float(val: str, default: float = float("nan")) -> float:
        try:
            v = float(val)
            return v if math.isfinite(v) else default
        except (ValueError, TypeError):
            return default

    with scores_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            if ds not in target_ds:
                continue
            sp = row["sample"]
            mt = row["method"]
            parsed_metrics = {
                k: _safe_float(row.get(k, ""))
                for k in ("ARI", "NMI", "FM", "PAS", "RAS", "CHAOS", "SI", "SI_X", "PAS_X", "RAS_X")
            }
            by_sample_method[(ds, sp)][mt] = parsed_metrics
            if mt not in ("gt", "consensus"):
                raw_rows.append({"dataset": ds, "sample": sp, "method": mt, **parsed_metrics})

    d_obs_map: Dict[str, Dict[str, float]] = {}
    d_ref_map: Dict[str, Dict[str, float]] = {}

    for (ds, sp), m_dict in sorted(by_sample_method.items()):
        sample_uid = f"{ds}::{sp}"
        spec = DATASET_TECH_SPECS.get(ds, {})
        sc_row = m_dict.get("scanpy", {})
        gt_row = m_dict.get("gt", {})

        # Label-free observable features
        d_obs_map[sample_uid] = {
            "is_single_cell": float(spec.get("is_single_cell", 0.5)),
            "log10_panel_genes": round(math.log10(max(10.0, float(spec.get("approx_genes", 1000.0)))), 4),
            "scanpy_PAS_spatial": round(sc_row.get("PAS", 0.35), 4) if math.isfinite(sc_row.get("PAS", float("nan"))) else 0.35,
            "scanpy_CHAOS_spatial": round(sc_row.get("CHAOS", 0.15), 4) if math.isfinite(sc_row.get("CHAOS", float("nan"))) else 0.15,
            "scanpy_SI_expr": round(sc_row.get("SI_X", 0.10), 4) if math.isfinite(sc_row.get("SI_X", float("nan"))) else 0.10,
        }
        # Ground-truth diagnostic features
        d_ref_map[sample_uid] = {
            "gt_PAS_spatial": round(gt_row.get("PAS", 0.05), 4) if math.isfinite(gt_row.get("PAS", float("nan"))) else 0.05,
            "gt_RAS_spatial": round(gt_row.get("RAS", 0.15), 4) if math.isfinite(gt_row.get("RAS", float("nan"))) else 0.15,
            "gt_SI_expr": round(gt_row.get("SI_X", 0.0), 4) if math.isfinite(gt_row.get("SI_X", float("nan"))) else 0.0,
            "gt_CHAOS_spatial": round(gt_row.get("CHAOS", 0.05), 4) if math.isfinite(gt_row.get("CHAOS", float("nan"))) else 0.05,
        }

    return raw_rows, d_obs_map, d_ref_map


def export_h5ad_to_canzar_3csv(
    h5ad_path: str | Path,
    dataset_name: str,
    sample_name: str,
    label_col: str = "ground_truth",
    framework_dir: Path = DEFAULT_FRAMEWORK_DIR,
) -> Path:
    """Convert any AnnData `.h5ad` file into Canzar Lab's `resources/<dataset>/<sample>_*.csv` format."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    adata = ad.read_h5ad(h5ad_path)
    valid = adata.obs[label_col].notna()
    adata = adata[valid].copy()

    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    coords = adata.obsm["spatial"][:, :2]
    labels = adata.obs[label_col]

    out_dir = framework_dir / "resources" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X).to_csv(out_dir / f"{sample_name}_UMImatrix.csv", index=False, header=False)
    pd.DataFrame(coords).to_csv(out_dir / f"{sample_name}_coords.csv", index=False, header=False)
    pd.Series(labels).to_csv(out_dir / f"{sample_name}_labels.csv", index=False, header=False)
    return out_dir


def generate_canzar_semisynthetic_slice(
    output_prefix: str | Path,
    shape: str = "stripes",
    perturbation: str = "none",
    dose: float = 0.0,
    seed: int = 1,
    simulator_dir: Path = DEFAULT_SIMULATOR_DIR,
) -> Tuple[Path, Path, Path]:
    """Invoke Canzar Lab's official `spatial-domain-simulator` (`subset_anndata.h5ad.gz`) on server."""
    import anndata as ad
    import pandas as pd

    out_prefix = Path(output_prefix)
    u_p = Path(f"{out_prefix}_UMImatrix.csv")
    c_p = Path(f"{out_prefix}_coords.csv")
    l_p = Path(f"{out_prefix}_labels.csv")
    if u_p.exists() and c_p.exists() and l_p.exists():
        return u_p, c_p, l_p

    if str(simulator_dir) not in sys.path:
        sys.path.insert(0, str(simulator_dir))

    import define_experiments as de  # type: ignore[import-not-found]

    gz_path = simulator_dir / "subset_anndata.h5ad.gz"
    unzipped_path = simulator_dir / "subset_anndata.h5ad"
    if unzipped_path.exists() and unzipped_path.stat().st_size < 100_000_000:
        # Remove truncated leftover from any interrupted gunzip
        unzipped_path.unlink()
    if not unzipped_path.exists():
        tmp_path = simulator_dir / "subset_anndata.h5ad.tmp"
        try:
            with gzip.open(gz_path, "rb") as f_in, tmp_path.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            tmp_path.replace(unzipped_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    adata = ad.read_h5ad(unzipped_path)
    celltypes_df = pd.read_csv(simulator_dir / "celltypes.csv", index_col=0)
    region_col = next(
        (c for c in ("ClusterNm", "Annotation", "cell_type", "region") if c in adata.obs.columns),
        adata.obs.columns[0],
    )

    if perturbation == "type1_ambient":
        counts, coords, labels = de.counts_perturbation_type1(
            adata, region_col, celltypes_df, shape=shape, noise_proportion=dose, cell_assignment_seed=seed
        )
    elif perturbation == "type2_infiltration":
        counts, coords, labels = de.counts_perturbation_type2(
            adata, region_col, celltypes_df, shape=shape, noise_proportion=dose, cell_assignment_seed=seed
        )
    elif perturbation == "thickness":
        div_right = int(round(30 + max(5.0, dose)))
        counts, coords, labels = de.vary_thickness(
            adata, region_col, celltypes_df, divides=(30, div_right), cell_assignment_seed=seed
        )
    else:
        counts, coords, labels = de.generate_blank_experiment(
            adata, region_col, celltypes_df, shape=shape, cell_assignment_seed=seed
        )
        if perturbation == "resolution_grid" and dose > 0.5:
            counts, coords, labels = de.modify_resolution(counts, coords, labels, gridlength=dose)
        elif perturbation == "gene_downsample" and 0.0 < dose < 1.0:
            counts, coords, labels = de.modify_number_of_genes(counts, coords, labels, downsample_prop=dose)
        elif perturbation == "sparsity" and dose >= 0.84:
            counts, coords, labels = de.modify_sparsity(counts, coords, labels, goalprop=dose)

    out_prefix = Path(output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    de.save_to_files(counts, coords, labels, str(out_prefix))
    return (
        Path(f"{out_prefix}_UMImatrix.csv"),
        Path(f"{out_prefix}_coords.csv"),
        Path(f"{out_prefix}_labels.csv"),
    )
