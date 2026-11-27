"""Agent 3: D-M Interaction & Counterfactual Reasoning Agent (v2.1 Audited).

Fixes from static code review:
1. Strictly withholds `heldout_markers` genes via `slc.get_model_input_counts()` before
   passing expression matrices to ANY method, making `Marker_Enrichment` genuinely leak-free.
2. Generates and saves a pre-confirmation experiment priority queue (`pre_confirmation_queue`)
   ranked strictly by outer-loop screening weights BEFORE touching confirmation slices, and
   evaluates active selection yield against random and fixed-order baselines along that exact queue.
3. Records real per-run `out.converged` rates and enforces hard validity & convergence gating
   inside donor-clustered confirmatory inference (`slice_to_donor`).
4. Upgrades preprocessing comparison to a crossed `Data Dose x Preprocessing Scheme (PCA-20 vs HVG-110)`
   factorial check (`tau_int_prep`), testing directly whether `D x C` explains dose sensitivity.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np

from factor_demo.agents.dataset_agent import DatasetCharacterAgent
from factor_demo.agents.method_agent import MethodCharacterAgent
from factor_demo.data_lab.generator_and_loader import SpatialSlice
from factor_demo.data_lab.perturbations import OPERATOR_SPECS, apply_data_operator
from factor_demo.llm_client import DualModeLLMClient
from factor_demo.methods_zoo.registry import METHODS_REGISTRY
from factor_demo.schemas import (
    DatasetProfile,
    FactorialContrastResult,
    MethodProfile,
    PreRegisteredHypothesis,
    ScreeningCandidate,
    ValidityAuditRecord,
)
from factor_demo.tools.eval_metrics import evaluate_predictions
from factor_demo.tools.stats_engine import (
    compute_factorial_inference,
    fit_outer_loop_screening,
)

SYSTEM_PROMPT = (
    "You are the D-M Interaction & Counterfactual Agent. Given observational screening results "
    "and controlled 2x3 factorial experimental contrasts on held-out confirmation donors, write a "
    "rigorous, non-overclaiming scientific synthesis. Distinguish operator-specific factorial "
    "interactions from main effects, note whether results hold after matched-budget hyperparameter "
    "retuning, and report crossed Dose x Preprocessing (`PCA-20 vs HVG-110`) findings."
)


class DMInteractionAgent:
    """Coordinates observational screening, pre-registered factorial interventions, and evaluation."""

    def __init__(
        self,
        llm_client: DualModeLLMClient,
        dataset_agent: DatasetCharacterAgent,
        method_agent: MethodCharacterAgent,
        n_pert_seeds: int = 2,
        n_null_seeds: int = 2,
        n_train_seeds: int = 3,
        equivalence_margin: float = 0.03,
        alpha: float = 0.05,
    ) -> None:
        self.llm = llm_client
        self.dataset_agent = dataset_agent
        self.method_agent = method_agent
        self.n_pert_seeds = n_pert_seeds
        self.n_null_seeds = n_null_seeds
        self.n_train_seeds = n_train_seeds
        self.equivalence_margin = equivalence_margin
        self.alpha = alpha
        self.pre_confirmation_queue: List[Dict[str, Any]] = []

    def run_observational_benchmark(
        self,
        slices: List[SpatialSlice],
        method_ids: List[str],
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, Dict[str, float]]]]:
        """Evaluate default methods across slices using strictly marker-withheld input counts."""
        N = len(slices)
        K = len(method_ids)
        y_ari = np.zeros((N, K), dtype=np.float64)
        detailed: Dict[str, Dict[str, Dict[str, float]]] = {}

        for i, slc in enumerate(slices):
            detailed[slc.slice_id] = {}
            x_train = slc.get_model_input_counts()
            for j, mid in enumerate(method_ids):
                runner = METHODS_REGISTRY[mid]["runner"]  # type: ignore[index]
                seed_metrics: List[Dict[str, float]] = []
                for s in range(self.n_train_seeds):
                    out = runner(x_train, slc.coords, slc.n_domains, arm="orig", train_seed=s)  # type: ignore[misc]
                    m = evaluate_predictions(
                        y_true=slc.domain_labels,
                        y_pred=out.pred_labels,
                        coords=slc.coords,
                        counts=slc.counts,  # Full counts used ONLY here to score withheld markers!
                        heldout_markers=slc.heldout_markers,
                    )
                    seed_metrics.append(m)

                avg_m = {
                    k: round(float(np.mean([sm[k] for sm in seed_metrics])), 4)
                    for k in seed_metrics[0]
                }
                avg_m["ARI_sd"] = round(float(np.std([sm["ARI"] for sm in seed_metrics])), 4)
                detailed[slc.slice_id][mid] = avg_m
                y_ari[i, j] = avg_m["ARI"]

        return y_ari, detailed

    def propose_and_preregister_hypotheses(
        self,
        expl_profiles: List[DatasetProfile],
        expl_slices: List[SpatialSlice],
        method_profiles: List[MethodProfile],
        y_expl: np.ndarray,
    ) -> Tuple[List[ScreeningCandidate], List[PreRegisteredHypothesis]]:
        """Run outer-loop screening on `D_expl` and build pre-confirmation priority queue BEFORE touching `D_conf`."""
        D_mat, d_names = self.dataset_agent.build_feature_matrix(expl_profiles, include_ref=True)
        M_mat, m_names, C_mat, c_names = self.method_agent.build_method_matrices(method_profiles)
        donor_ids = [s.donor_id for s in expl_slices]

        candidates = fit_outer_loop_screening(
            d_matrix=D_mat,
            d_names=d_names,
            m_matrix=M_mat,
            m_names=m_names,
            c_matrix=C_mat,
            c_names=c_names,
            y_obs=y_expl,
            donor_ids=donor_ids,
        )

        top_dm = next((c for c in candidates if c.feature_group == "D_x_M"), candidates[0])
        top_dc = next((c for c in candidates if c.feature_group == "D_x_C"), candidates[-1])
        method_ids = [m.method_id for m in method_profiles]

        h1 = PreRegisteredHypothesis(
            hypothesis_id="H1_Infiltration_x_GraphTopology",
            title="Operational Interaction: Cross-Boundary Cell Infiltration x Spatial Graph Topology",
            rationale_from_screening=(
                f"Outer-loop CV screening surfaced `{top_dm.d_feature} x {top_dm.m_or_c_feature}` "
                f"(W={top_dm.estimate_w:+.4f}, stability={top_dm.stability_prob:.2f}). "
                f"Tests how boundary infiltration alters real spatial topology gain (`refac - null`)."
            ),
            target_task="multicellular_spatial_domain_segmentation",
            data_operator_name="u2_boundary_infiltration",
            data_operator_doses=list(OPERATOR_SPECS["u2_boundary_infiltration"]["default_doses"]),  # type: ignore[arg-type]
            label_evolution_rule="fixed_anatomical_region_geometry",
            target_methods=method_ids,
            method_arms=["orig", "refac", "self", "null"],
            competing_pipeline_factor="Dose x Preprocessing_Scheme (PCA-20 vs HVG-110)",
            tuning_protocols=["fixed", "retuned"],
            primary_metric="ARI",
            equivalence_margin=self.equivalence_margin,
            expected_direction="Unpruned multi-hop smoothing loses topological advantage under high boundary infiltration.",
            falsification_rule=(
                f"Practically equivalent if 90% TOST CI on independent confirmation donors stays within "
                f"[-{self.equivalence_margin}, +{self.equivalence_margin}]."
            ),
        )

        h2 = PreRegisteredHypothesis(
            hypothesis_id="H2_Resolution_x_GraphTopology",
            title="Operational Interaction: Spot Spatial Aggregation x Spatial Graph Topology",
            rationale_from_screening=(
                f"Coupled with `{top_dc.d_feature} x {top_dc.m_or_c_feature}` "
                f"(W={top_dc.estimate_w:+.4f}, stability={top_dc.stability_prob:.2f}): "
                f"tests how coarsening single-cell coordinates into multi-cell spots shifts spatial topology gain."
            ),
            target_task="multicellular_spatial_domain_segmentation",
            data_operator_name="u1_spot_aggregation",
            data_operator_doses=[1.0, 6.0],
            label_evolution_rule="majority_underlying_niche_vote",
            target_methods=method_ids,
            method_arms=["orig", "refac", "self", "null"],
            competing_pipeline_factor="Dose x Preprocessing_Scheme (PCA-20 vs HVG-110)",
            tuning_protocols=["fixed"],
            primary_metric="ARI",
            equivalence_margin=self.equivalence_margin,
            expected_direction="Spot aggregation alters relative benefit of local spatial graph smoothing.",
            falsification_rule=(
                f"Practically equivalent if 90% TOST CI on confirmation donors stays within "
                f"[-{self.equivalence_margin}, +{self.equivalence_margin}]."
            ),
        )

        # Build genuine pre-confirmation active experiment priority queue BEFORE seeing confirmation slices!
        op_target_features = {
            "H1_Infiltration_x_GraphTopology": {"local_expr_heterogeneity", "global_morans_i", "boundary_edge_ratio_ref", "min_domain_fraction_ref"},
            "H2_Resolution_x_GraphTopology": {"spatial_density", "sparsity_fraction", "mean_genes_per_unit"},
        }
        m_map = {mp.method_id: {**mp.m_core, **mp.c_aux} for mp in method_profiles}
        queue: List[Dict[str, Any]] = []
        fixed_idx = 0
        for hyp in (h1, h2):
            rel_d = op_target_features.get(hyp.hypothesis_id, set())
            for mid in method_ids:
                fixed_idx += 1
                m_flags = m_map[mid]
                # Sum outer-loop priority scores over screening terms matching this (operator, method) pair
                pred_info_score = sum(
                    c.priority_score
                    for c in candidates
                    if c.d_feature in rel_d and m_flags.get(c.m_or_c_feature, 0.0) > 0.5
                )
                # Also include residual variance of method on exploration set as uncertainty bonus
                j_idx = method_ids.index(mid)
                expl_var = float(np.std(y_expl[:, j_idx]))
                total_acq_score = pred_info_score + 0.15 * expl_var
                queue.append(
                    {
                        "hypothesis_id": hyp.hypothesis_id,
                        "method_id": mid,
                        "fixed_order_rank": fixed_idx,
                        "predicted_acquisition_score": round(float(total_acq_score), 5),
                    }
                )

        queue.sort(key=lambda item: item["predicted_acquisition_score"], reverse=True)
        for rank_i, item in enumerate(queue, start=1):
            item["active_priority_rank"] = rank_i
        self.pre_confirmation_queue = queue

        return candidates, [h1, h2]

    def execute_confirmatory_experiments(
        self,
        hypotheses: List[PreRegisteredHypothesis],
        conf_slices: List[SpatialSlice],
        stagewise_gate_by_method: Optional[Dict[str, bool]] = None,
        budget_k: Optional[int] = None,
    ) -> Tuple[
        List[FactorialContrastResult],
        List[ValidityAuditRecord],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        """Run pre-registered 2x3 factorial experiments + donor-level `Dose x Preprocessing` on `D_conf`.

        When `budget_k` is provided, executes ONLY the top `budget_k` pre-registered hypotheses
        ranked in `self.pre_confirmation_queue` (online compute-saving execution mode!).
        """
        all_contrasts: List[FactorialContrastResult] = []
        audit_records: List[ValidityAuditRecord] = []
        dose_curve_store: Dict[str, Any] = {}
        slice_to_donor = {s.slice_id: s.donor_id for s in conf_slices}
        gate_map = stagewise_gate_by_method or {}

        selected_hypotheses = list(hypotheses)
        if budget_k is not None and budget_k > 0 and self.pre_confirmation_queue:
            rank_order = {
                str(item["hypothesis_id"]): int(item["priority_rank"])
                for item in self.pre_confirmation_queue
            }
            selected_hypotheses.sort(key=lambda h: rank_order.get(h.hypothesis_id, 999))
            selected_hypotheses = selected_hypotheses[:budget_k]

        for hyp in selected_hypotheses:
            op_name = hyp.data_operator_name
            doses = hyp.data_operator_doses
            dose_low, dose_high = float(doses[0]), float(doses[-1])
            dose_curve_store[hyp.hypothesis_id] = {"doses": doses, "methods": {}}

            # 1. Pre-flight equivalence & drift diagnostics per slice and dose
            op_validity_by_method: Dict[str, List[bool]] = {m: [] for m in hyp.target_methods}
            temp_audits: List[ValidityAuditRecord] = []

            for slc in conf_slices:
                eq_diffs = self.method_agent.verify_code_equivalence(slc, hyp.target_methods, seed=0)
                for d_val in doses:
                    pert_slc = apply_data_operator(slc, operator_name=op_name, dose=float(d_val), seed=0)
                    drift = self.dataset_agent.audit_counterfactual_slice(slc, pert_slc, operator_name=op_name)
                    # For invariant features of this operator, threshold off-target z-drift <= 1.25
                    drift_ok = float(drift["max_off_target_drift_z"]) <= 1.25
                    for mid in hyp.target_methods:
                        stage_ok = bool(gate_map.get(mid, True))
                        op_validity_by_method[mid].append(bool(stage_ok and eq_diffs[mid] <= 1e-4 and drift_ok))

                    temp_audits.append(
                        ValidityAuditRecord(
                            slice_id=slc.slice_id,
                            operator_name=op_name,
                            dose=float(d_val),
                            target_shift_z=float(drift["max_target_shift_z"]),
                            max_off_target_drift_z=float(drift["max_off_target_drift_z"]),
                            off_target_details=dict(drift["off_target_drifts_z"]),  # type: ignore[arg-type]
                            gene_corr_frobenius_drift=float(drift["gene_corr_frobenius_drift"]),
                            code_equiv_max_diff=eq_diffs,
                            convergence_ok_rate=0.0,  # Populated below from actual runs!
                            passed_validity_check=bool(max(eq_diffs.values()) <= 1e-4 and drift_ok),
                        )
                    )

            # 2. Execute full factorial arms & record REAL convergence booleans
            all_conv_flags_for_op: List[bool] = []
            for mid in hyp.target_methods:
                runner = METHODS_REGISTRY[mid]["runner"]  # type: ignore[index]
                dose_curve_store[hyp.hypothesis_id]["methods"][mid] = {}
                validity_ok = all(op_validity_by_method[mid])

                for proto in hyp.tuning_protocols:
                    per_slice_cells: Dict[str, Dict[str, float]] = {s.slice_id: {} for s in conf_slices}
                    curve_means: Dict[str, List[float]] = {a: [] for a in ("refac", "self", "null")}
                    method_conv_flags: List[bool] = []

                    for d_val in doses:
                        for arm in ("refac", "self", "null"):
                            slice_scores = []
                            for slc in conf_slices:
                                run_vals = []
                                p_seeds = [0] if d_val == doses[0] else list(range(self.n_pert_seeds))
                                g_seeds = list(range(self.n_null_seeds)) if arm == "null" else [0]

                                for ps in p_seeds:
                                    pert_slc = apply_data_operator(slc, operator_name=op_name, dose=float(d_val), seed=ps)
                                    x_in = pert_slc.get_model_input_counts()
                                    for gs in g_seeds:
                                        for ts in range(self.n_train_seeds):
                                            out = runner(  # type: ignore[misc]
                                                x_in,
                                                pert_slc.coords,
                                                pert_slc.n_domains,
                                                arm=arm,
                                                tuning_protocol=proto,
                                                null_seed=gs + 100 * ps,
                                                train_seed=ts,
                                            )
                                            method_conv_flags.append(bool(out.converged))
                                            all_conv_flags_for_op.append(bool(out.converged))
                                            ev = evaluate_predictions(
                                                y_true=pert_slc.domain_labels,
                                                y_pred=out.pred_labels,
                                                coords=pert_slc.coords,
                                                counts=pert_slc.counts,
                                                heldout_markers=pert_slc.heldout_markers,
                                            )
                                            run_vals.append(ev[hyp.primary_metric])

                                mean_s = float(np.mean(run_vals))
                                per_slice_cells[slc.slice_id][f"{float(d_val)}_{arm}"] = mean_s
                                slice_scores.append(mean_s)

                            curve_means[arm].append(round(float(np.mean(slice_scores)), 4))

                    dose_curve_store[hyp.hypothesis_id]["methods"][mid][proto] = curve_means
                    conv_rate = float(np.mean(method_conv_flags)) if method_conv_flags else 0.0

                    contrast = compute_factorial_inference(
                        hypothesis_id=hyp.hypothesis_id,
                        method_id=mid,
                        tuning_protocol=proto,
                        operator_name=op_name,
                        dose_low=dose_low,
                        dose_high=dose_high,
                        metric_name=hyp.primary_metric,
                        per_slice_cells=per_slice_cells,
                        slice_to_donor=slice_to_donor,
                        equivalence_margin=hyp.equivalence_margin,
                        alpha=self.alpha,
                        validity_passed=validity_ok,
                        convergence_rate=conv_rate,
                    )
                    all_contrasts.append(contrast)

            op_conv_rate = round(float(np.mean(all_conv_flags_for_op)), 4) if all_conv_flags_for_op else 0.0
            for ar in temp_audits:
                ar.convergence_ok_rate = op_conv_rate
                ar.passed_validity_check = bool(ar.passed_validity_check and op_conv_rate >= 0.80)
            audit_records.extend(temp_audits)

            if len(hyp.tuning_protocols) >= 2:
                for mid in hyp.target_methods:
                    pair = [c for c in all_contrasts if c.hypothesis_id == hyp.hypothesis_id and c.method_id == mid]
                    if len(pair) == 2:
                        c_fix, c_ret = pair[0], pair[1]
                        if (
                            c_fix.evidence_state.startswith("Supported")
                            and c_ret.evidence_state.startswith("Practically Equivalent")
                        ) or (c_fix.tau_int_topo_mean * c_ret.tau_int_topo_mean < -0.001):
                            c_fix.evidence_state += " [Protocol-Sensitive]"
                            c_ret.evidence_state += " [Protocol-Sensitive]"

        # 3. Crossed `Data Dose (low vs high) x Preprocessing Scheme (PCA-20 vs HVG-110)` with Donor-Level CI & TOST!
        from scipy import stats as sp_stats

        h1 = hypotheses[0]
        d_lo, d_hi = float(h1.data_operator_doses[0]), float(h1.data_operator_doses[-1])
        prep_crossover: Dict[str, Dict[str, Any]] = {}

        for mid in h1.target_methods:
            runner = METHODS_REGISTRY[mid]["runner"]  # type: ignore[index]
            pca_lo_all, hvg_lo_all, pca_hi_all, hvg_hi_all = [], [], [], []
            donor_tau_buckets: Dict[str, List[float]] = {}

            for slc in conf_slices:
                s_lo = apply_data_operator(slc, h1.data_operator_name, dose=d_lo, seed=0)
                s_hi = apply_data_operator(slc, h1.data_operator_name, dose=d_hi, seed=0)
                x_lo = s_lo.get_model_input_counts()
                x_hi = s_hi.get_model_input_counts()
                s_pca_lo, s_hvg_lo, s_pca_hi, s_hvg_hi = [], [], [], []
                for ts in range(self.n_train_seeds):
                    o_pca_lo = runner(x_lo, s_lo.coords, s_lo.n_domains, arm="refac", preprocess_override="pca", train_seed=ts)  # type: ignore[misc]
                    o_hvg_lo = runner(x_lo, s_lo.coords, s_lo.n_domains, arm="refac", preprocess_override="hvg", train_seed=ts)  # type: ignore[misc]
                    o_pca_hi = runner(x_hi, s_hi.coords, s_hi.n_domains, arm="refac", preprocess_override="pca", train_seed=ts)  # type: ignore[misc]
                    o_hvg_hi = runner(x_hi, s_hi.coords, s_hi.n_domains, arm="refac", preprocess_override="hvg", train_seed=ts)  # type: ignore[misc]

                    s_pca_lo.append(evaluate_predictions(s_lo.domain_labels, o_pca_lo.pred_labels, s_lo.coords, s_lo.counts, s_lo.heldout_markers)["ARI"])
                    s_hvg_lo.append(evaluate_predictions(s_lo.domain_labels, o_hvg_lo.pred_labels, s_lo.coords, s_lo.counts, s_lo.heldout_markers)["ARI"])
                    s_pca_hi.append(evaluate_predictions(s_hi.domain_labels, o_pca_hi.pred_labels, s_hi.coords, s_hi.counts, s_hi.heldout_markers)["ARI"])
                    s_hvg_hi.append(evaluate_predictions(s_hi.domain_labels, o_hvg_hi.pred_labels, s_hi.coords, s_hi.counts, s_hi.heldout_markers)["ARI"])

                pca_lo_all.extend(s_pca_lo)
                hvg_lo_all.extend(s_hvg_lo)
                pca_hi_all.extend(s_pca_hi)
                hvg_hi_all.extend(s_hvg_hi)

                slc_diff_lo = float(np.mean(s_pca_lo) - np.mean(s_hvg_lo))
                slc_diff_hi = float(np.mean(s_pca_hi) - np.mean(s_hvg_hi))
                donor_tau_buckets.setdefault(slc.donor_id, []).append(slc_diff_hi - slc_diff_lo)

            donor_taus = np.array([float(np.mean(v)) for v in donor_tau_buckets.values()], dtype=np.float64)
            n_donors_prep = len(donor_taus)
            mean_tau_prep = float(np.mean(donor_taus))
            eq_m = h1.equivalence_margin

            if n_donors_prep < 2:
                ci95_lo, ci95_hi = float("-inf"), float("inf")
                prep_state = "INSUFFICIENT_DONOR_REPLICATES"
            else:
                se_prep = float(np.std(donor_taus, ddof=1) / np.sqrt(n_donors_prep))
                if se_prep < 1e-10:
                    ci95_lo, ci95_hi = mean_tau_prep, mean_tau_prep
                    ci90_lo, ci90_hi = mean_tau_prep, mean_tau_prep
                else:
                    t95 = float(sp_stats.t.ppf(1.0 - self.alpha / 2.0, df=n_donors_prep - 1))
                    t90 = float(sp_stats.t.ppf(1.0 - self.alpha, df=n_donors_prep - 1))
                    ci95_lo, ci95_hi = mean_tau_prep - t95 * se_prep, mean_tau_prep + t95 * se_prep
                    ci90_lo, ci90_hi = mean_tau_prep - t90 * se_prep, mean_tau_prep + t90 * se_prep

                if ci95_lo > eq_m:
                    prep_state = "Supported Positive Prep x Dose Interaction"
                elif ci95_hi < -eq_m:
                    prep_state = "Supported Negative Prep x Dose Interaction"
                elif ci90_lo >= -eq_m and ci90_hi <= eq_m:
                    prep_state = "Practically Equivalent (within +/-Delta_eq)"
                else:
                    prep_state = "Inconclusive"

            diff_lo = float(np.mean(pca_lo_all) - np.mean(hvg_lo_all))
            diff_hi = float(np.mean(pca_hi_all) - np.mean(hvg_hi_all))
            prep_crossover[mid] = {
                "ARI_with_PCA": round(float(0.5 * (np.mean(pca_lo_all) + np.mean(pca_hi_all))), 4),
                "ARI_with_HVG": round(float(0.5 * (np.mean(hvg_lo_all) + np.mean(hvg_hi_all))), 4),
                "PCA_minus_HVG": round(0.5 * (diff_lo + diff_hi), 4),
                "PCA_minus_HVG_dose_low": round(diff_lo, 4),
                "PCA_minus_HVG_dose_high": round(diff_hi, 4),
                "tau_int_prep_scheme": round(mean_tau_prep, 4),
                "tau_int_prep_ci95_low": round(ci95_lo, 4) if np.isfinite(ci95_lo) else -999.0,
                "tau_int_prep_ci95_high": round(ci95_hi, 4) if np.isfinite(ci95_hi) else 999.0,
                "n_donors": n_donors_prep,
                "prep_evidence_state": prep_state,
            }

        return all_contrasts, audit_records, dose_curve_store, prep_crossover

    def evaluate_agent_and_recommendation_value(
        self,
        expl_profiles: List[DatasetProfile],
        conf_profiles: List[DatasetProfile],
        y_expl: np.ndarray,
        y_conf: np.ndarray,
        method_ids: List[str],
        candidates: List[ScreeningCandidate],
        contrasts: List[FactorialContrastResult],
    ) -> Dict[str, Any]:
        """Evaluate (1) Label-free out-of-sample recommendation regret & (2) Pre-registered Active Queue Yield."""
        from sklearn.linear_model import RidgeCV

        X_train, _ = self.dataset_agent.build_feature_matrix(expl_profiles, include_ref=False)
        X_test, _ = self.dataset_agent.build_feature_matrix(conf_profiles, include_ref=False)

        mu = np.mean(X_train, axis=0, keepdims=True)
        sd = np.std(X_train, axis=0, keepdims=True) + 1e-6
        X_tr_s = (X_train - mu) / sd
        X_te_s = (X_test - mu) / sd

        preds = np.zeros_like(y_conf)
        for j in range(len(method_ids)):
            reg = RidgeCV(alphas=np.logspace(-1, 2, 7))
            reg.fit(X_tr_s, y_expl[:, j])
            preds[:, j] = reg.predict(X_te_s)

        global_best_j = int(np.argmax(np.mean(y_expl, axis=0)))
        oracle_best = np.max(y_conf, axis=1)

        agent_chosen_j = np.argmax(preds, axis=1)
        agent_realized = np.array([y_conf[i, agent_chosen_j[i]] for i in range(len(conf_profiles))])
        static_realized = y_conf[:, global_best_j]

        regret_agent = float(np.mean(oracle_best - agent_realized))
        regret_static = float(np.mean(oracle_best - static_realized))

        # Genuine Active Queue Evaluation: map `(hypothesis_id, method_id)` along `self.pre_confirmation_queue`!
        fixed_map = {
            (c.hypothesis_id, c.method_id): (
                1 if ("Supported" in c.evidence_state or "Statistically Non-Zero" in c.evidence_state) else 0
            )
            for c in contrasts
            if c.tuning_protocol == "fixed"
        }
        total_tests = len(fixed_map)
        total_supported = sum(fixed_map.values())
        budget_k = min(4, total_tests)

        active_ordered_keys = [
            (item["hypothesis_id"], item["method_id"])
            for item in self.pre_confirmation_queue
            if (item["hypothesis_id"], item["method_id"]) in fixed_map
        ]
        fixed_ordered_keys = [
            (item["hypothesis_id"], item["method_id"])
            for item in sorted(self.pre_confirmation_queue, key=lambda x: x["fixed_order_rank"])
            if (item["hypothesis_id"], item["method_id"]) in fixed_map
        ]

        active_hits = sum(fixed_map[k] for k in active_ordered_keys[:budget_k])
        fixed_order_hits = sum(fixed_map[k] for k in fixed_ordered_keys[:budget_k])
        expected_random_hits = budget_k * (total_supported / max(1, total_tests))

        return {
            "global_best_method_on_expl": method_ids[global_best_j],
            "mean_oracle_ari_conf": round(float(np.mean(oracle_best)), 4),
            "mean_static_winner_ari_conf": round(float(np.mean(static_realized)), 4),
            "mean_d_obs_recommended_ari_conf": round(float(np.mean(agent_realized)), 4),
            "regret_static_winner": round(regret_static, 4),
            "regret_d_obs_recommender": round(regret_agent, 4),
            "per_slice_recommendations": [
                {
                    "slice_id": conf_profiles[i].dataset_id,
                    "recommended": method_ids[int(agent_chosen_j[i])],
                    "oracle_best": method_ids[int(np.argmax(y_conf[i]))],
                    "realized_ari": round(float(agent_realized[i]), 4),
                    "oracle_ari": round(float(oracle_best[i]), 4),
                }
                for i in range(len(conf_profiles))
            ],
            "pre_confirmation_queue": self.pre_confirmation_queue,
            "active_budget_k": budget_k,
            "active_supported_discoveries": int(active_hits),
            "fixed_order_discoveries": int(fixed_order_hits),
            "random_baseline_expected_discoveries": round(float(expected_random_hits), 2),
            "full_grid_total_tests": int(total_tests),
        }

    def synthesize_scientific_conclusion(
        self,
        candidates: List[ScreeningCandidate],
        contrasts: List[FactorialContrastResult],
        prep_crossover: Dict[str, Dict[str, float]],
        system_eval: Dict[str, Any],
    ) -> str:
        """Generate final calibrated scientific synthesis grounded in all deterministic outputs."""
        payload = {
            "top_screening_candidates": [c.__dict__ for c in candidates[:5]],
            "confirmatory_contrasts": [c.to_dict() for c in contrasts if c.tuning_protocol == "fixed"],
            "crossed_dose_x_preprocessing": prep_crossover,
            "system_evaluation": system_eval,
        }
        user_prompt = json.dumps(payload, ensure_ascii=False)

        def _offline_synthesis() -> str:
            h1_rows = [
                c for c in contrasts
                if c.hypothesis_id == "H1_Infiltration_x_GraphTopology" and c.tuning_protocol == "fixed"
            ]
            lines = [
                "1. **Operator-Specific Bilateral Interaction (`u2_boundary_infiltration x Graph Topology`)**: "
                "Across independent confirmation donors, increasing cross-boundary cellular infiltration from 0% to 50% "
                "alters topological advantage (`refac - null`):"
            ]
            for r in h1_rows:
                lines.append(
                    f"   - `{r.method_id}`: simple topological effect shifts from {r.simple_topo_low:+.3f} (0% swap) "
                    f"to {r.simple_topo_high:+.3f} (50% swap), donor-clustered interaction "
                    f"tau_int_topo = {r.tau_int_topo_mean:+.3f} (95% CI [{r.tau_int_topo_ci_low:+.3f}, {r.tau_int_topo_ci_high:+.3f}], "
                    f"status: **{r.evidence_state}**)."
                )
            lines.append(
                "2. **Crossed `Dose x Preprocessing Scheme (PCA-20 vs. HVG-110)` Test**: "
                + ", ".join(
                    f"`{m}` (mean PCA-HVG={v['PCA_minus_HVG']:+.3f}, dose interaction tau_prep={v['tau_int_prep_scheme']:+.3f})"
                    for m, v in prep_crossover.items()
                )
                + " — directly quantifying how much regime shift comes from preprocessing scheme vs. graph topology."
            )
            lines.append(
                f"3. **Out-of-Sample & Active Queue Metrics**: Label-free `d_obs` recommender regret = "
                f"{system_eval['regret_d_obs_recommender']:.4f} (vs. static winner {system_eval['regret_static_winner']:.4f}); "
                f"pre-registered active priority queue hit {system_eval['active_supported_discoveries']}/{system_eval['active_budget_k']} "
                f"non-zero interactions (vs. random expectation {system_eval['random_baseline_expected_discoveries']:.2f})."
            )
            return "\n".join(lines)

        return self.llm.complete(
            agent_name="DMInteractionAgent",
            stage="final_scientific_synthesis",
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            offline_fallback_fn=_offline_synthesis,
        )
