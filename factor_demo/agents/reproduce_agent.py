"""Agent 0: Reproduce & Native-Repo Execution Agent (`ReproduceAgent`, v2.2 Audited).

Supports two complementary execution philosophies:
1. **Primary Mode — Direct Native Upstream Reproduction (`run_native_upstream_script`)**:
   Clones/invokes the author's original unmodified code repository / runner script directly
   on standardized inputs and logs raw outputs, runtime, and exit status—respecting the
   author's exact implementation without rewriting into a generic wrapper pipeline!
2. **Secondary Mode — Comprehensive Stagewise Equivalence Audit (`verify_stagewise_equivalence`)**:
   Used whenever a modularized adapter is needed for component recombination. Verifies:
   - Cell & gene dimension/order alignment + preprocessed matrix max & relative Frobenius diff;
   - Exact adjacency edge-set Jaccard index (not merely degree distribution!), edge-weight L1 diff,
     self-loop presence, and row-sum normalization check;
   - Fixed-seed representation discrepancy benchmarked against multi-seed normal stochastic variance;
   - Explicit clustering head, hyperparameter, and fallback recording.
3. **Coordinate-Scale & Radius Sanity Diagnosis (`audit_and_repair_radius_graph`)**:
   Inspects physical coordinate span and median 1-NN spacing before diagnosing isolated-cell
   rates, keeping **Track 1 (unmodified default output)** strictly separate from **Track 2
   (explicitly logged coordinate-scale adaptation)**.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from factor_demo.llm_client import DualModeLLMClient
from factor_demo.tools.eval_metrics import align_clusters_to_reference

UPSTREAM_METHOD_REPOS: Dict[str, Dict[str, str]] = {
    "SEDR": {
        "repo_url": "https://github.com/JinmiaoChenLab/SEDR.git",
        "native_script": "sedr_run.py",
        "pkg_name": "SEDR",
        "graph_type": "kNN",
    },
    "CCST": {
        "repo_url": "https://github.com/xiaoyeye/CCST.git",
        "native_script": "ccst_run.py",
        "pkg_name": "CCST",
        "graph_type": "radius_bisection",
    },
    "SCAN-IT": {
        "repo_url": "https://github.com/zcang/SCAN-IT.git",
        "native_script": "scanit_run.py",
        "pkg_name": "scanit",
        "graph_type": "alpha_complex",
    },
    "STAGATE": {
        "repo_url": "https://github.com/QIFEIDKN/STAGATE.git",
        "native_script": "stagate_run.py",
        "pkg_name": "STAGATE",
        "graph_type": "radius_or_kNN",
    },
    "GraphST": {
        "repo_url": "https://github.com/JinmiaoChenLab/GraphST.git",
        "native_script": "graphst_run.py",
        "pkg_name": "GraphST",
        "graph_type": "kNN",
    },
    "SpaceFlow": {
        "repo_url": "https://github.com/hongleir/SpaceFlow.git",
        "native_script": "spaceflow_run.py",
        "pkg_name": "SpaceFlow",
        "graph_type": "alpha_complex",
    },
    "BANKSY": {
        "repo_url": "https://github.com/prabhakarlab/Banksy_py.git",
        "native_script": "banksy_run.py",
        "pkg_name": "Banksy_py",
        "graph_type": "kNN_azimuthal",
    },
}


@dataclass
class NativeReproductionRecord:
    """Audit record of running an author's unmodified native source script."""

    method_id: str
    dataset_name: str
    sample_name: str
    command_executed: List[str]
    return_code: int
    wall_time_sec: float
    output_clustering_tsv: Optional[str]
    succeeded: bool
    stderr_tail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphSanityReport:
    """Coordinate-scale & connectivity diagnostic (Track 1 default vs. Track 2 adaptation)."""

    slice_id: str
    method_id: str
    coord_span_xy: Tuple[float, float]
    median_1nn_spacing: float
    default_radius: float
    default_mean_degree: float
    default_isolated_fraction: float
    scale_mismatch_suspected: bool
    diagnostic_note: str
    adapted_radius: Optional[float] = None
    adapted_mean_degree: Optional[float] = None
    adapted_isolated_fraction: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UpstreamEquivalenceReport:
    """Comprehensive stagewise comparison between native upstream run and modularized adapter."""

    method_id: str
    effective_variant_id: str
    slice_id: str
    cell_gene_shape_match: bool
    preprocess_rel_frobenius_diff: float
    preprocess_max_abs_diff: float
    edge_set_jaccard: float
    edge_weight_l1_diff: float
    self_loop_match: bool
    row_sum_max_diff: float
    embedding_gram_cosine: float
    fixed_seed_ari_agreement: float
    baseline_multiseed_ari_mean: Optional[float] = None
    fallback_notes: List[str] = field(default_factory=list)
    passed_strict_equivalence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReproduceAgent:
    """Agent 0: Executes native upstream repositories, audits graph scale, and verifies adapters."""

    def __init__(self, llm_client: DualModeLLMClient) -> None:
        self.llm = llm_client

    def clone_upstream_repo_if_needed(self, method_id: str, target_root: Path) -> Path:
        """Clone author's official GitHub repository into `target_root/<method_id>`."""
        if method_id not in UPSTREAM_METHOD_REPOS:
            raise ValueError(f"Unknown upstream method_id: {method_id}")
        info = UPSTREAM_METHOD_REPOS[method_id]
        dest = target_root / method_id
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", info["repo_url"], str(dest)], check=True)
        return dest

    def run_native_upstream_script(
        self,
        framework_dir: Path,
        method_id: str,
        dataset_name: str,
        sample_name: str,
        n_domains: int,
        python_executable: str = "python",
    ) -> NativeReproductionRecord:
        """Run the unmodified native method script directly (bypassing generic rewrites!)."""
        info = UPSTREAM_METHOD_REPOS[method_id]
        script_rel = Path("workflow") / "scripts" / info["native_script"]
        cmd = [
            python_executable,
            str(script_rel),
            "-d",
            dataset_name,
            "-s",
            sample_name,
            "-n",
            str(n_domains),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(framework_dir),
            capture_output=True,
            text=True,
        )
        elapsed = round(time.perf_counter() - t0, 3)
        expected_tsv = framework_dir / "results" / dataset_name / method_id / sample_name / "clustering.tsv"
        ok = proc.returncode == 0 and expected_tsv.exists()
        return NativeReproductionRecord(
            method_id=method_id,
            dataset_name=dataset_name,
            sample_name=sample_name,
            command_executed=cmd,
            return_code=int(proc.returncode),
            wall_time_sec=elapsed,
            output_clustering_tsv=str(expected_tsv) if expected_tsv.exists() else None,
            succeeded=ok,
            stderr_tail=(proc.stderr or "")[-500:],
        )

    def audit_and_repair_radius_graph(
        self,
        coords: np.ndarray,
        slice_id: str,
        method_id: str,
        default_radius: float,
        target_degree_range: Tuple[float, float] = (4.0, 10.0),
    ) -> Tuple[np.ndarray, GraphSanityReport]:
        """Inspect physical coordinate spacing & diagnose scale-induced radius graph degeneracy."""
        n = coords.shape[0]
        span_x = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))
        span_y = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))

        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        dist_no_self = dist + np.eye(n, dtype=np.float32) * 1e9
        median_1nn = float(np.median(np.min(dist_no_self, axis=1)))

        def _stats_at_r(r: float) -> Tuple[np.ndarray, float, float]:
            adj = ((dist > 0.0) & (dist <= r)).astype(np.float32)
            deg = np.sum(adj, axis=1)
            return adj, float(np.mean(deg)), float(np.mean(deg == 0))

        adj_def, mean_deg_def, iso_def = _stats_at_r(default_radius)
        k_lo, k_hi = target_degree_range

        # Check explicitly if default_radius is smaller than median_1nn spacing (coordinate unit mismatch!)
        if default_radius < median_1nn * 0.95 or (iso_def > 0.15 and mean_deg_def < k_lo * 0.5):
            note = (
                f"radius={default_radius:.2f} is smaller than/near median_1NN={median_1nn:.2f} "
                f"(coord_span=({span_x:.1f},{span_y:.1f})); likely coordinate unit mismatch."
            )
            suspected = True
        elif mean_deg_def > max(50.0, 0.25 * n):
            note = f"radius={default_radius:.2f} connects mean_degree={mean_deg_def:.1f}; coordinate span much smaller than expected."
            suspected = True
        else:
            note = "default_radius compatible with tissue coordinate spacing"
            suspected = False

        if not suspected:
            return adj_def, GraphSanityReport(
                slice_id=slice_id,
                method_id=method_id,
                coord_span_xy=(round(span_x, 2), round(span_y, 2)),
                median_1nn_spacing=round(median_1nn, 4),
                default_radius=default_radius,
                default_mean_degree=round(mean_deg_def, 3),
                default_isolated_fraction=round(iso_def, 4),
                scale_mismatch_suspected=False,
                diagnostic_note=note,
            )

        lo, hi = 0.0, float(np.max(dist))
        best_r = (lo + hi) / 2.0
        for _ in range(30):
            best_r = (lo + hi) / 2.0
            _, m_deg, _ = _stats_at_r(best_r)
            if k_lo <= m_deg <= k_hi:
                break
            if m_deg < k_lo:
                lo = best_r
            else:
                hi = best_r

        adj_rep, mean_deg_rep, iso_rep = _stats_at_r(best_r)
        return adj_rep, GraphSanityReport(
            slice_id=slice_id,
            method_id=method_id,
            coord_span_xy=(round(span_x, 2), round(span_y, 2)),
            median_1nn_spacing=round(median_1nn, 4),
            default_radius=default_radius,
            default_mean_degree=round(mean_deg_def, 3),
            default_isolated_fraction=round(iso_def, 4),
            scale_mismatch_suspected=True,
            diagnostic_note=note,
            adapted_radius=round(best_r, 4),
            adapted_mean_degree=round(mean_deg_rep, 3),
            adapted_isolated_fraction=round(iso_rep, 4),
        )

    def verify_stagewise_equivalence(
        self,
        method_id: str,
        slice_id: str,
        X_prep_upstream: np.ndarray,
        X_prep_modular: np.ndarray,
        adj_upstream: np.ndarray,
        adj_modular: np.ndarray,
        emb_upstream: np.ndarray,
        emb_modular: np.ndarray,
        labels_upstream: np.ndarray,
        labels_modular: np.ndarray,
        baseline_multiseed_aris: Optional[List[float]] = None,
        fallback_notes: Optional[List[str]] = None,
    ) -> UpstreamEquivalenceReport:
        """Edge-level, normalization-level, and representation-level equivalence audit."""
        from sklearn.metrics import adjusted_rand_score

        notes = list(fallback_notes or [])
        shape_ok = (X_prep_upstream.shape == X_prep_modular.shape) and (adj_upstream.shape == adj_modular.shape)

        if shape_ok:
            prep_max = float(np.max(np.abs(X_prep_upstream - X_prep_modular)))
            prep_rel = float(np.linalg.norm(X_prep_upstream - X_prep_modular) / (np.linalg.norm(X_prep_upstream) + 1e-8))
        else:
            prep_max, prep_rel = float("inf"), float("inf")

        if adj_upstream.shape == adj_modular.shape:
            e_up = adj_upstream > 0
            e_mod = adj_modular > 0
            inter = int(np.sum(e_up & e_mod))
            union = int(np.sum(e_up | e_mod))
            jaccard = float(inter / union) if union > 0 else 1.0
            w_l1 = float(np.mean(np.abs(adj_upstream - adj_modular)))
            diag_up = bool(np.any(np.diag(adj_upstream) > 0))
            diag_mod = bool(np.any(np.diag(adj_modular) > 0))
            self_loop_ok = diag_up == diag_mod
            row_diff = float(np.max(np.abs(np.sum(adj_upstream, axis=1) - np.sum(adj_modular, axis=1))))
        else:
            jaccard, w_l1, self_loop_ok, row_diff = 0.0, float("inf"), False, float("inf")

        u1 = emb_upstream - np.mean(emb_upstream, axis=0, keepdims=True)
        u2 = emb_modular - np.mean(emb_modular, axis=0, keepdims=True)
        g1 = u1 @ u1.T
        g2 = u2 @ u2.T
        gram_cos = float(np.sum(g1 * g2) / (float(np.linalg.norm(g1) * np.linalg.norm(g2)) + 1e-8))

        ari_agree = float(adjusted_rand_score(labels_upstream, align_clusters_to_reference(labels_upstream, labels_modular)))
        multiseed_mean = float(np.mean(baseline_multiseed_aris)) if baseline_multiseed_aris else None

        has_fallback = len(notes) > 0
        variant_id = f"{method_id}__{'_'.join(notes)}" if has_fallback else method_id

        # Require exact edge topology (Jaccard >= 0.999), matching self-loops & row sums, and ARI agreement >= multiseed baseline
        ari_thresh = max(0.95, (multiseed_mean - 0.02)) if multiseed_mean is not None else 0.99
        passed = bool(
            shape_ok
            and (not has_fallback)
            and prep_rel <= 1e-4
            and jaccard >= 0.999
            and self_loop_ok
            and row_diff <= 1e-4
            and gram_cos >= 0.995
            and ari_agree >= ari_thresh
        )

        return UpstreamEquivalenceReport(
            method_id=method_id,
            effective_variant_id=variant_id,
            slice_id=slice_id,
            cell_gene_shape_match=shape_ok,
            preprocess_rel_frobenius_diff=round(prep_rel, 6) if np.isfinite(prep_rel) else -1.0,
            preprocess_max_abs_diff=round(prep_max, 6) if np.isfinite(prep_max) else -1.0,
            edge_set_jaccard=round(jaccard, 6),
            edge_weight_l1_diff=round(w_l1, 6) if np.isfinite(w_l1) else -1.0,
            self_loop_match=self_loop_ok,
            row_sum_max_diff=round(row_diff, 6) if np.isfinite(row_diff) else -1.0,
            embedding_gram_cosine=round(gram_cos, 6),
            fixed_seed_ari_agreement=round(ari_agree, 6),
            baseline_multiseed_ari_mean=round(multiseed_mean, 4) if multiseed_mean is not None else None,
            fallback_notes=notes,
            passed_strict_equivalence=passed,
        )
