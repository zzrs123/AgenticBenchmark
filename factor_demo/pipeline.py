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
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the full 3-Agent workflow and export JSON artifacts, PNG figures, Markdown, & PDF."""
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
        print(f"[FACTOR Demo] Starting 3-Agent pipeline | Mode: {mode_banner}")

    # 1. Load real .h5ad cohort if provided, else build realistic 12-slice cohort (8 Expl + 4 Conf)
    if h5ad_dir is not None:
        h5_files = sorted(Path(h5ad_dir).glob("*.h5ad"))
        if len(h5_files) < 4:
            raise ValueError(f"Expected at least 4 .h5ad files in {h5ad_dir}, found {len(h5_files)}")
        n_expl = max(2, int(round(len(h5_files) * 0.67)))
        all_slices: List[SpatialSlice] = [
            load_h5ad_slice(f, split="exploration" if idx < n_expl else "confirmation")
            for idx, f in enumerate(h5_files)
        ]
    else:
        all_slices = build_demo_cohort(seed=42)

    expl_slices = [s for s in all_slices if s.split == "exploration"]
    conf_slices = [s for s in all_slices if s.split == "confirmation"]
    ext_test_slices = [s for s in all_slices if s.split == "external_test"]
    if not ext_test_slices:
        ext_test_slices = conf_slices
    method_ids = list(METHODS_REGISTRY.keys())

    # 2. Agent 1: Dataset-Character Agent
    if verbose:
        print(f"[Agent 1: DatasetCharacterAgent] Profiling {len(all_slices)} spatial slices...")
    dataset_agent = DatasetCharacterAgent(llm_client=llm)
    all_d_profiles = dataset_agent.profile_cohort(all_slices)
    expl_profiles = [p for p in all_d_profiles if p.split == "exploration"]
    conf_profiles = [p for p in all_d_profiles if p.split == "confirmation"]
    ext_test_profiles = [p for p in all_d_profiles if p.split == "external_test"] or conf_profiles

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

    # Agent 0: ReproduceAgent radius-scale sanity check on confirmation slices
    reproduce_agent = ReproduceAgent(llm_client=llm)
    graph_sanity_logs = [
        reproduce_agent.audit_and_repair_radius_graph(s.coords, s.slice_id, "RadiusGraphCheck", default_radius=8.0)[1].to_dict()
        for s in conf_slices
    ]

    if verbose:
        print("[Agent 3: Inner Loop] Executing controlled 2x3 factorial experiments on held-out Confirmation donors...")
    y_conf, obs_detail_conf = interaction_agent.run_observational_benchmark(conf_slices, method_ids)
    contrasts, audit_records, dose_curve_store, prep_crossover = (
        interaction_agent.execute_confirmatory_experiments(hypotheses, conf_slices)
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

    # Evaluate the 3 Downstream Deliverables strictly on UNTOUCHED External Test Split (`ext_test_slices`)!
    import time
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

    deliverables_eval = {
        "evaluation_split": "external_test (untouched during screening & confirmation)",
        "n_external_test_donors": len(ext_test_slices),
        "chimera_4way_ablation_mean_ARI": {
            k: round(float(np.mean(v)), 4) for k, v in chimera_ablation_aris.items()
        },
        "ensemble_d_obs_weighted_mean_ARI": round(float(np.mean(ensemble_weighted_aris)), 4),
        "ensemble_uniform_mean_ARI": round(float(np.mean(ensemble_uniform_aris)), 4),
        "recommender_d_obs_mean_ARI": round(float(np.mean(rec_ext_aris)), 4),
        "static_expl_winner_mean_ARI": round(float(np.mean(static_ext_aris)), 4),
        "total_wall_time_sec": deliv_wall_sec,
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
