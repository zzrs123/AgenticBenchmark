"""Agent 1: Dataset-Character Agent.

Responsibilities:
1. Invoke deterministic spatial/transcriptional metric tools on input tissue slices.
2. Isolate label-free observable features (`d_obs`) from reference-dependent diagnostics (`d_ref`).
3. Provide closed-loop Intervention Validity & Off-Target Drift Diagnostics when Agent 3
   applies operational perturbations (`u1_spot_aggregation`, `u2_boundary_infiltration`).
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from factor_demo.data_lab.generator_and_loader import SpatialSlice
from factor_demo.data_lab.perturbations import OPERATOR_SPECS
from factor_demo.llm_client import DualModeLLMClient
from factor_demo.schemas import DatasetProfile
from factor_demo.tools.dataset_metrics import (
    audit_perturbation_drift,
    compute_dataset_features,
)

SYSTEM_PROMPT = (
    "You are the Dataset-Character Agent in a spatial transcriptomics meta-benchmark system. "
    "Summarize the spatial slice concisely in 2 sentences strictly using the provided quantitative "
    "tool metrics (d_obs and d_ref). Never invent numbers; highlight sparsity, spatial smoothness "
    "(Moran's I / local heterogeneity), and potential domain boundary challenges."
)


class DatasetCharacterAgent:
    """Extracts dataset meta-features and audits counterfactual slice perturbations."""

    def __init__(self, llm_client: DualModeLLMClient) -> None:
        self.llm = llm_client
        self.feature_scales: Dict[str, float] = {}

    def profile_cohort(self, slices: List[SpatialSlice]) -> List[DatasetProfile]:
        """Characterize a cohort of spatial slices and calibrate feature scales on exploration data."""
        profiles: List[DatasetProfile] = []
        raw_records: List[Dict[str, float]] = []

        for slc in slices:
            trace = [f"call:compute_dataset_features(slice_id='{slc.slice_id}')"]
            d_obs, d_ref = compute_dataset_features(slc)
            if slc.split == "exploration":
                raw_records.append({**d_obs, **d_ref})

            user_prompt = json.dumps(
                {
                    "slice_id": slc.slice_id,
                    "platform": slc.platform_tag,
                    "tissue": slc.tissue_tag,
                    "d_obs": d_obs,
                    "d_ref": d_ref,
                },
                ensure_ascii=False,
            )

            def _offline_summary(
                sid: str = slc.slice_id,
                plat: str = slc.platform_tag,
                tis: str = slc.tissue_tag,
                obs: Dict[str, float] = d_obs,
                ref: Dict[str, float] = d_ref,
            ) -> str:
                het_tag = (
                    "high local heterogeneity / boundary mixing"
                    if obs["local_expr_heterogeneity"] >= 0.27
                    else "smooth laminar spatial organization"
                )
                thin_tag = (
                    f"contains narrow domain (min fraction={ref['min_domain_fraction_ref']:.2f})"
                    if ref["min_domain_fraction_ref"] < 0.16
                    else f"balanced domain sizes (min fraction={ref['min_domain_fraction_ref']:.2f})"
                )
                return (
                    f"[{plat} | {tis}] {sid} exhibits {het_tag} "
                    f"(Moran's I={obs['global_morans_i']:.3f}, local_het={obs['local_expr_heterogeneity']:.3f}, "
                    f"sparsity={obs['sparsity_fraction']:.2%}); {thin_tag}."
                )

            summary = self.llm.complete(
                agent_name="DatasetCharacterAgent",
                stage=f"profile_{slc.slice_id}",
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                offline_fallback_fn=_offline_summary,
            )
            trace.append("synthesized_slice_diagnostic_card")

            profiles.append(
                DatasetProfile(
                    dataset_id=slc.slice_id,
                    split=slc.split,
                    platform_tag=slc.platform_tag,
                    tissue_tag=slc.tissue_tag,
                    n_units=int(slc.counts.shape[0]),
                    n_genes=int(slc.counts.shape[1]),
                    n_domains=int(slc.n_domains),
                    d_obs=d_obs,
                    d_ref=d_ref,
                    agent_summary=summary,
                    tool_trace=trace,
                )
            )

        # Compute empirical SD on exploration split for standardized drift diagnostics
        if raw_records:
            keys = list(raw_records[0].keys())
            for k in keys:
                vals = np.array([r[k] for r in raw_records], dtype=np.float64)
                self.feature_scales[k] = max(float(np.std(vals)), 1e-3)

        return profiles

    def build_feature_matrix(
        self,
        profiles: List[DatasetProfile],
        include_ref: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """Stack standardized descriptor vectors across profiles."""
        if not profiles:
            raise ValueError("Empty profiles list")
        obs_keys = sorted(profiles[0].d_obs.keys())
        ref_keys = sorted(profiles[0].d_ref.keys()) if include_ref else []
        all_keys = obs_keys + ref_keys

        rows = []
        for p in profiles:
            merged = {**p.d_obs, **p.d_ref}
            rows.append([float(merged[k]) for k in all_keys])
        return np.asarray(rows, dtype=np.float64), all_keys

    def audit_counterfactual_slice(
        self,
        orig_slice: SpatialSlice,
        pert_slice: SpatialSlice,
        operator_name: str,
    ) -> Dict[str, object]:
        """Run Intervention Validity & Off-Target Drift Diagnostics on a perturbed slice."""
        spec = OPERATOR_SPECS[operator_name]
        scales = self.feature_scales or {
            "local_expr_heterogeneity": 0.05,
            "global_morans_i": 0.06,
            "sparsity_fraction": 0.15,
            "mean_genes_per_unit": 35.0,
            "spatial_density": 0.12,
            "n_profiled_genes": 40.0,
        }
        return audit_perturbation_drift(
            orig_slice=orig_slice,
            pert_slice=pert_slice,
            feature_scales=scales,
            target_features=list(spec["target_features"]),  # type: ignore[arg-type]
            invariant_features=list(spec["invariant_features"]),  # type: ignore[arg-type]
        )
