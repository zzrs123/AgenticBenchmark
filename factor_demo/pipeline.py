"""End-to-end orchestrator uniting Agent 1, Agent 2, Agent 3, Visualizations, and Reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from factor_demo.agents.dataset_agent import DatasetCharacterAgent
from factor_demo.agents.interaction_agent import DMInteractionAgent
from factor_demo.agents.method_agent import MethodCharacterAgent
from factor_demo.agents.reproduce_agent import ReproduceAgent
from factor_demo.deliverables import (
    FactorAuditedChimera,
    FactorMethodRecommender,
    run_label_free_consensus,
)
from factor_demo.tools.eval_metrics import evaluate_predictions
from factor_demo.data_lab.generator_and_loader import (
    SpatialSlice,
    assign_donor_grouped_splits,
    build_demo_cohort,
    load_h5ad_slice,
)
from factor_demo.llm_client import DualModeLLMClient
from factor_demo.methods_zoo.registry import METHODS_REGISTRY
from factor_demo.reporting.report_builder import (
    generate_markdown_report,
    generate_pdf_report,
)
from factor_demo.reporting.visualizer import (
    plot_fig1_observational_and_screening,
    plot_fig2_factorial_dose_curves,
    plot_fig3_tissue_map_case_study,
    plot_fig4_forest_and_competing_checks,
)


def run_demo_pipeline(
    output_dir: str | Path = "outputs/demo_run",
    llm_mode: str = "auto",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    h5ad_dir: Optional[str | Path] = None,
    n_pert_seeds: int = 2,
    n_null_seeds: int = 2,
    n_train_seeds: int = 3,
    equivalence_margin: float = 0.03,
    budget_k: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the full 4-Agent workflow and export JSON artifacts, PNG figures, Markdown, & PDF."""
    out_root = Path(output_dir).resolve()
    fig_dir = out_root / "figures"
    json_dir = out_root / "artifacts"
    fig_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    llm = DualModeLLMClient(
        mode=llm_mode,
        base_url=base_url,
        api_key=api_key,
        model=model_name,
    )
    mode_banner = (
        f"Live OpenAI-Compatible API ({llm.model} @ {llm.base_url})"
        if llm.is_live_available
        else "Offline Deterministic Tool-Grounded Trace Mode"
    )
    if verbose:
        print(f"[FACTOR Demo] Starting 4-Agent pipeline | Mode: {mode_banner}")

    # 1. Load real .h5ad cohort (strictly donor-grouped!) if provided, else build 15-slice cohort (8 Expl + 4 Conf + 3 ExtTest)
    if h5ad_dir is not None:
        h5_files = sorted(Path(h5ad_dir).glob("*.h5ad"))
        if len(h5_files) < 4:
            raise ValueError(f"Expected at least 4 .h5ad files in {h5ad_dir}, found {len(h5_files)}")
        raw_slices: List[SpatialSlice] = [load_h5ad_slice(f, split=None) for f in h5_files]
        all_slices = assign_donor_grouped_splits(raw_slices)
    else:
        all_slices = assign_donor_grouped_splits(build_demo_cohort(seed=42))

    expl_slices = [s for s in all_slices if s.split == "exploration"]
    conf_slices = [s for s in all_slices if s.split == "confirmation"]
    ext_test_slices = [s for s in all_slices if s.split == "external_test"]
    method_ids = list(METHODS_REGISTRY.keys())

    # 2. Agent 1: Dataset-Character Agent (strictly using marker-excluded `get_model_input_counts()`)
    if verbose:
        print(f"[Agent 1: DatasetCharacterAgent] Profiling {len(all_slices)} spatial slices...")
    dataset_agent = DatasetCharacterAgent(llm_client=llm)
    all_d_profiles = dataset_agent.profile_cohort(all_slices)
    expl_profiles = [p for p in all_d_profiles if p.split == "exploration"]
    conf_profiles = [p for p in all_d_profiles if p.split == "confirmation"]
    ext_test_profiles = [p for p in all_d_profiles if p.split == "external_test"]

    # 3. Agent 2: Method-Character Agent
    if verbose:
        print(f"[Agent 2: MethodCharacterAgent] Parsing Python ASTs for {len(method_ids)} methods...")
    method_agent = MethodCharacterAgent(llm_client=llm)
    method_profiles = method_agent.profile_methods(method_ids)

    # 4. Agent 3: D-M Interaction & Counterfactual Agent
    interaction_agent = DMInteractionAgent(
        llm_client=llm,
        dataset_agent=dataset_agent,
        method_agent=method_agent,
        n_pert_seeds=n_pert_seeds,
        n_null_seeds=n_null_seeds,
        n_train_seeds=n_train_seeds,
        equivalence_margin=equivalence_margin,
    )

    if verbose:
        print("[Agent 3: Outer Loop] Running observational benchmark & identifiable PLMM screening...")
    y_expl, obs_detail_expl = interaction_agent.run_observational_benchmark(expl_slices, method_ids)
    candidates, hypotheses = interaction_agent.propose_and_preregister_hypotheses(
        expl_profiles=expl_profiles,
        expl_slices=expl_slices,
        method_profiles=method_profiles,
        y_expl=y_expl,
    )

    # Save pre-registered hypotheses AND active experiment priority queue BEFORE running confirmation interventions
    (json_dir / "preregistered_hypotheses.json").write_text(
        json.dumps([h.to_dict() for h in hypotheses], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (json_dir / "pre_confirmation_experiment_queue.json").write_text(
        json.dumps(interaction_agent.pre_confirmation_queue, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Agent 0: ReproduceAgent radius-scale sanity check AND pre-flight stagewise equivalence gate
    from factor_demo.methods_zoo.base_pipeline import build_spatial_knn_graph, preprocess_expression

    reproduce_agent = ReproduceAgent(llm_client=llm)
    graph_sanity_logs = [
        reproduce_agent.audit_and_repair_radius_graph(s.coords, s.slice_id, "RadiusGraphCheck", default_radius=8.0)[1].to_dict()
        for s in conf_slices
    ]
    stagewise_eq_logs = []
    stagewise_gate_by_method: Dict[str, bool] = {mid: True for mid in method_ids}
    for slc in conf_slices:
        x_in = slc.get_model_input_counts()
        adj_check = build_spatial_knn_graph(slc.coords, k_neighbors=8, include_self=True)
        X_prep_check = preprocess_expression(x_in, mode="pca", n_hvgs=110, n_pcs=20, seed=0)
        for mid in method_ids:
            runner = METHODS_REGISTRY[mid]["runner"]  # type: ignore[index]
            out_orig = runner(x_in, slc.coords, slc.n_domains, arm="orig", train_seed=0)  # type: ignore[misc]
            out_refac = runner(x_in, slc.coords, slc.n_domains, arm="refac", train_seed=0)  # type: ignore[misc]
            eq_rep = reproduce_agent.verify_stagewise_equivalence(
                method_id=mid,
                slice_id=slc.slice_id,
                X_prep_upstream=X_prep_check,
                X_prep_modular=X_prep_check,
                adj_upstream=adj_check,
                adj_modular=adj_check,
                emb_upstream=out_orig.embedding,
                emb_modular=out_refac.embedding,
                labels_upstream=out_orig.pred_labels,
                labels_modular=out_refac.pred_labels,
            )
            stagewise_eq_logs.append(eq_rep.to_dict())
            if not eq_rep.passed_strict_equivalence:
                stagewise_gate_by_method[mid] = False

    (json_dir / "stagewise_equivalence_audit.json").write_text(
        json.dumps(stagewise_eq_logs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if verbose:
        print("[Agent 3: Inner Loop] Executing controlled 2x3 factorial experiments on held-out Confirmation donors...")
    y_conf, obs_detail_conf = interaction_agent.run_observational_benchmark(conf_slices, method_ids)
    contrasts, audit_records, dose_curve_store, prep_crossover = (
        interaction_agent.execute_confirmatory_experiments(
            hypotheses,
            conf_slices,
            stagewise_gate_by_method=stagewise_gate_by_method,
            budget_k=budget_k,
        )
    )

    system_eval = interaction_agent.evaluate_agent_and_recommendation_value(
        expl_profiles=expl_profiles,
        conf_profiles=conf_profiles,
        y_expl=y_expl,
        y_conf=y_conf,
        method_ids=method_ids,
        candidates=candidates,
        contrasts=contrasts,
    )

    agent_conclusion = interaction_agent.synthesize_scientific_conclusion(
        candidates=candidates,
        contrasts=contrasts,
        prep_crossover=prep_crossover,
        system_eval=system_eval,
    )

    # Evaluate the 3 Downstream Deliverables STRICTLY when untouched `external_test` donors exist!
    # NEVER silently substitute `conf_slices` as `ext_test_slices`!
    import time
    if len(ext_test_slices) > 0:
        chimera_method = FactorAuditedChimera()
        chimera_method.calibrate_on_exploration([p.d_obs for p in expl_profiles])
        recommender = FactorMethodRecommender()
        recommender.fit([p.d_obs for p in expl_profiles], y_expl=y_expl, method_ids=method_ids)

        chimera_ablation_aris: Dict[str, List[float]] = {v: [] for v in FactorAuditedChimera.VARIANTS}
        ensemble_weighted_aris, ensemble_uniform_aris = [], []
        rec_ext_aris, static_ext_aris = [], []
        global_best_mid = method_ids[int(np.argmax(np.mean(y_expl, axis=0)))]

        t_deliv_start = time.perf_counter()
        for idx_t, slc in enumerate(ext_test_slices):
            x_in = slc.get_model_input_counts()
            d_obs_t = ext_test_profiles[idx_t].d_obs

            for var_name in FactorAuditedChimera.VARIANTS:
                c_out = chimera_method.run(x_in, slc.coords, slc.n_domains, d_obs_slice=d_obs_t, variant=var_name, seed=0)
                chimera_ablation_aris[var_name].append(
                    evaluate_predictions(slc.domain_labels, c_out.pred_labels, slc.coords, slc.counts, slc.heldout_markers)["ARI"]
                )

            preds_each = {
                mid: METHODS_REGISTRY[mid]["runner"](x_in, slc.coords, slc.n_domains, arm="orig", train_seed=0).pred_labels  # type: ignore[index,misc]
                for mid in method_ids
            }
            each_ari = {
                mid: evaluate_predictions(slc.domain_labels, preds_each[mid], slc.coords, slc.counts, slc.heldout_markers)["ARI"]
                for mid in method_ids
            }
            rec_choice = recommender.recommend(d_obs_t)["recommended_method"]
            rec_ext_aris.append(each_ari[rec_choice])
            static_ext_aris.append(each_ari[global_best_mid])

            w_pred = recommender.predict_scores(d_obs_t)
            ens_w = run_label_free_consensus(preds_each, n_domains=slc.n_domains, method_weights=w_pred, seed=0)
            ens_u = run_label_free_consensus(preds_each, n_domains=slc.n_domains, method_weights=None, seed=0)
            ensemble_weighted_aris.append(
                evaluate_predictions(slc.domain_labels, ens_w, slc.coords, slc.counts, slc.heldout_markers)["ARI"]
            )
            ensemble_uniform_aris.append(
                evaluate_predictions(slc.domain_labels, ens_u, slc.coords, slc.counts, slc.heldout_markers)["ARI"]
            )
        deliv_wall_sec = round(time.perf_counter() - t_deliv_start, 3)

        lambda_control_audit = chimera_method.audit_lambda_weight_control(
            counts=ext_test_slices[0].get_model_input_counts(),
            coords=ext_test_slices[0].coords,
            n_domains=ext_test_slices[0].n_domains,
            seed=0,
        )

        deliverables_eval = {
            "evaluation_split": "external_test (donor-disjoint, untouched during screening & confirmation)",
            "n_external_test_donors": len({s.donor_id for s in ext_test_slices}),
            "n_external_test_slices": len(ext_test_slices),
            "chimera_4way_ablation_mean_ARI": {
                k: round(float(np.mean(v)), 4) for k, v in chimera_ablation_aris.items()
            },
            "chimera_kmeans_4way_ablation_mean_ARI": {
                var_name: round(
                    float(
                        np.mean(
                            [
                                evaluate_predictions(
                                    slc.domain_labels,
                                    chimera_method.run(
                                        slc.get_model_input_counts(),
                                        slc.coords,
                                        slc.n_domains,
                                        d_obs_slice=ext_test_profiles[i_s].d_obs,
                                        variant=var_name,
                                        cluster_head="kmeans",
                                        seed=0,
                                    ).pred_labels,
                                    slc.coords,
                                    slc.counts,
                                    slc.heldout_markers,
                                )["ARI"]
                                for i_s, slc in enumerate(ext_test_slices)
                            ]
                        )
                    ),
                    4,
                )
                for var_name in FactorAuditedChimera.VARIANTS
            },
            "chimera_lambda_weight_control_audit": lambda_control_audit,
            "ensemble_d_obs_weighted_mean_ARI": round(float(np.mean(ensemble_weighted_aris)), 4),
            "ensemble_uniform_mean_ARI": round(float(np.mean(ensemble_uniform_aris)), 4),
            "recommender_d_obs_mean_ARI": round(float(np.mean(rec_ext_aris)), 4),
            "static_expl_winner_mean_ARI": round(float(np.mean(static_ext_aris)), 4),
            "total_wall_time_sec": deliv_wall_sec,
        }
    else:
        deliverables_eval = {
            "evaluation_split": "SKIPPED_NO_UNTOUCHED_EXTERNAL_TEST_DONORS",
            "n_external_test_donors": 0,
            "n_external_test_slices": 0,
            "note": "Cohort has <6 distinct donors; confirmation slices are never reused as external_test.",
        }

    # 5. Generate Figures
    if verbose:
        print("[Reporting] Rendering publication figures, Markdown report, and multi-page PDF...")
    y_all = np.vstack([y_expl, y_conf])
    slice_ids_ordered = [s.slice_id for s in expl_slices + conf_slices]
    splits_ordered = [s.split for s in expl_slices + conf_slices]

    rep_slice = next((s for s in conf_slices if "ThinLayer" in s.slice_id), conf_slices[0])
    fig_paths = {
        "fig1": plot_fig1_observational_and_screening(
            slice_ids=slice_ids_ordered,
            splits=splits_ordered,
            method_ids=method_ids,
            y_all=y_all,
            candidates=candidates,
            out_path=fig_dir / "fig1_observational_screening.png",
        ),
        "fig2": plot_fig2_factorial_dose_curves(
            dose_curve_store=dose_curve_store,
            hypothesis_id="H1_Infiltration_x_GraphTopology",
            method_ids=method_ids,
            out_path=fig_dir / "fig2_factorial_2x3_curves.png",
        ),
        "fig3": plot_fig3_tissue_map_case_study(
            rep_slice=rep_slice,
            out_path=fig_dir / "fig3_tissue_map_case_study.png",
        ),
        "fig4": plot_fig4_forest_and_competing_checks(
            contrasts=contrasts,
            prep_crossover=prep_crossover,
            system_eval=system_eval,
            out_path=fig_dir / "fig4_forest_and_competing_checks.png",
        ),
    }

    md_report = generate_markdown_report(
        output_dir=out_root,
        llm_mode_summary=mode_banner,
        dataset_profiles=all_d_profiles,
        method_profiles=method_profiles,
        candidates=candidates,
        hypotheses=hypotheses,
        audit_records=audit_records,
        contrasts=contrasts,
        prep_crossover=prep_crossover,
        system_eval=system_eval,
        agent_conclusion=agent_conclusion,
        figure_paths=fig_paths,
    )
    pdf_report = generate_pdf_report(
        output_dir=out_root,
        llm_mode_summary=mode_banner,
        contrasts=contrasts,
        system_eval=system_eval,
        figure_paths=fig_paths,
    )

    # Persist full JSON trace bundle for auditability
    summary_bundle = {
        "llm_mode": mode_banner,
        "dataset_profiles": [p.to_dict() for p in all_d_profiles],
        "method_profiles": [p.to_dict() for p in method_profiles],
        "screening_candidates": [c.__dict__ for c in candidates[:12]],
        "preregistered_hypotheses": [h.to_dict() for h in hypotheses],
        "confirmatory_contrasts": [c.to_dict() for c in contrasts],
        "preprocessing_crossover": prep_crossover,
        "system_evaluation": system_eval,
        "deliverables_evaluation": deliverables_eval,
        "reproduce_graph_sanity": graph_sanity_logs,
        "stagewise_equivalence_audit": stagewise_eq_logs,
        "llm_call_log": [r.__dict__ for r in llm.call_history],
    }
    (json_dir / "full_run_bundle.json").write_text(
        json.dumps(summary_bundle, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if verbose:
        print(f"[Done] Markdown Report: {md_report}")
        print(f"[Done] Multi-Page PDF : {pdf_report}")

    return {
        "markdown_report": md_report,
        "pdf_report": pdf_report,
        "figures": fig_paths,
        "bundle": summary_bundle,
    }
