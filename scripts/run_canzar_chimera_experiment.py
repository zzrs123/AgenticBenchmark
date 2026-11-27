#!/usr/bin/env python3
"""Server CLI to run live 4-arm counterfactuals on Canzar 2026 real/simulator slices & Chimera methods.

Includes mandatory pre-flight `ReproduceAgent` auditing:
1.Coordinate-scale & radius connectivity check (`audit_and_repair_radius_graph`);
2. Direct unmodified native upstream script execution (`run_native_upstream_script`) on `osmFISH_Codeluppi`
   plus stagewise adapter equivalence check (`verify_stagewise_equivalence`);
3. Full persistence of `effective_method_id`, `fallback_notes`, `sedr_intervention_mode`, and
   `upstream_gate_status` in `live_chimera_counterfactual_summary.json` (so `pyGMM` fallbacks or
   native divergences are NEVER silently conflated with verified upstream runs!).

Examples (inside `Chimera` or `scanpy+pytorch` Conda environment on server):
1. Run native audit + 4-arm (`orig`, `refac`, `self`, `null`) counterfactual on `osmFISH_Codeluppi`:
   python scripts/run_canzar_chimera_experiment.py --use-osmfish --methods GraphST SEDR scanpy_nonspatial

2. Generate official mouse-brain snRNA-seq semi-synthetic slices (`subset_anndata.h5ad.gz`) across
   Type II cellular infiltration doses and run 4-arm counterfactuals:
   python scripts/run_canzar_chimera_experiment.py --simulate-infiltration --methods GraphST STAGATE SEDR --sedr-mode encoder_only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_demo.agents.reproduce_agent import UPSTREAM_METHOD_REPOS, ReproduceAgent
from factor_demo.canzar_bridge.canzar_data_loader import (
    DEFAULT_FRAMEWORK_DIR,
    generate_canzar_semisynthetic_slice,
)
from factor_demo.canzar_bridge.chimera_counterfactual_driver import (
    run_chimera_counterfactual,
)
from factor_demo.llm_client import DualModeLLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live Canzar 2026 Chimera 4-arm counterfactuals with ReproduceAgent gating.")
    parser.add_argument("--use-osmfish", action="store_true", help="Use real osmFISH_Codeluppi slice in external repo.")
    parser.add_argument("--simulate-infiltration", action="store_true", help="Generate Type II infiltration doses from subset_anndata.h5ad.gz.")
    parser.add_argument("--doses", nargs="+", type=float, default=[0.0, 0.25, 0.50], help="Infiltration doses.")
    parser.add_argument("--methods", nargs="+", default=["GraphST", "SEDR", "scanpy_nonspatial"], help="Chimera modules to test.")
    parser.add_argument("--clust", choices=["mclust", "kmeans"], default="mclust")
    parser.add_argument(
        "--sedr-mode",
        choices=["encoder_only", "full_mechanism"],
        default="encoder_only",
        help="SEDR intervention scope: encoder propagation only vs encoder + graph reconstruction loss.",
    )
    parser.add_argument("--skip-native-gate", action="store_true", help="Skip pre-flight native upstream script execution.")
    parser.add_argument(
        "--strict-native-gate",
        action="store_true",
        help="Abort `self`/`null` counterfactual arms for any method failing native upstream agreement or triggering cluster fallback.",
    )
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "canzar_live_chimera"))
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    reproduce_agent = ReproduceAgent(llm_client=DualModeLLMClient(mode="offline"))

    records: List[Dict[str, Any]] = []
    native_audit_logs: List[Dict[str, Any]] = []
    stagewise_audit_logs: List[Dict[str, Any]] = []
    graph_sanity_logs: List[Dict[str, Any]] = []

    slice_tasks = []
    if args.use_osmfish or not args.simulate_infiltration:
        base = DEFAULT_FRAMEWORK_DIR / "resources" / "osmFISH_Codeluppi" / "correctedsample"
        slice_tasks.append(("osmFISH_Codeluppi", "correctedsample", 0.0, Path(f"{base}_UMImatrix.csv"), Path(f"{base}_coords.csv"), Path(f"{base}_labels.csv")))

    if args.simulate_infiltration:
        for dose in args.doses:
            sample_tag = f"stripes_type2_dose_{dose:.2f}"
            prefix = out_root / "sim_slices" / sample_tag
            u_csv, c_csv, l_csv = generate_canzar_semisynthetic_slice(
                output_prefix=prefix,
                shape="stripes",
                perturbation="type2_infiltration",
                dose=dose,
                seed=1,
            )
            slice_tasks.append(("SimMouseBrain_Type2", sample_tag, dose, u_csv, c_csv, l_csv))

    # Track method-level native upstream agreement from pre-flight anchor check
    native_agreement_by_method: Dict[str, Optional[float]] = {}

    for dataset_tag, sample_tag, dose, umi_csv, coords_csv, labels_csv in slice_tasks:
        y_true = pd.read_csv(labels_csv, header=None).iloc[:, 0].astype(str).to_numpy()
        coords_arr = pd.read_csv(coords_csv, header=None).to_numpy(dtype=np.float32)[:, :2]
        n_dom = len(np.unique(y_true))

        _, g_rep = reproduce_agent.audit_and_repair_radius_graph(
            coords=coords_arr,
            slice_id=f"{dataset_tag}::{sample_tag}",
            method_id="PreFlightCoordCheck",
            default_radius=150.0,
        )
        graph_sanity_logs.append(g_rep.to_dict())

        for mid in args.methods:
            # 1. Pre-flight native upstream script check on built-in `osmFISH_Codeluppi`
            native_labels: Optional[np.ndarray] = None
            if (not args.skip_native_gate) and dataset_tag == "osmFISH_Codeluppi" and mid in UPSTREAM_METHOD_REPOS:
                print(f"[Agent 0: ReproduceAgent] Running unmodified native script for {mid} on {dataset_tag}...")
                nat_rec = reproduce_agent.run_native_upstream_script(
                    framework_dir=DEFAULT_FRAMEWORK_DIR,
                    method_id=mid,
                    dataset_name=dataset_tag,
                    sample_name=sample_tag,
                    n_domains=n_dom,
                    python_executable=sys.executable,
                )
                native_audit_logs.append(nat_rec.to_dict())
                if nat_rec.succeeded and nat_rec.output_clustering_tsv:
                    nat_df = pd.read_csv(nat_rec.output_clustering_tsv, sep="\t")
                    col = "region" if "region" in nat_df.columns else nat_df.columns[-1]
                    raw_nat = nat_df[col].astype(str).to_numpy()
                    if len(raw_nat) == len(y_true):
                        _, inv_nat = np.unique(raw_nat, return_inverse=True)
                        native_labels = inv_nat.astype(np.int64)

            # 2. Run `orig` and `refac` first to audit internal & upstream equivalence BEFORE `self` / `null`!
            arm_outputs: Dict[str, Dict[str, Any]] = {}
            for pre_arm in ("orig", "refac"):
                print(f"[Pre-Flight Arm] {dataset_tag} (dose={dose}) | Method={mid} | Arm={pre_arm}")
                arm_outputs[pre_arm] = run_chimera_counterfactual(
                    umi_csv=umi_csv,
                    coords_csv=coords_csv,
                    n_domains=n_dom,
                    preprocess_mod=mid,
                    adjacency_mod=mid,
                    train_mod=mid,
                    clust_mod=args.clust,
                    arm=pre_arm,
                    sedr_intervention_mode=args.sedr_mode,
                    null_seed=0,
                    output_dir=out_root / dataset_tag / f"dose_{dose:.2f}" / mid / pre_arm,
                )

            res_orig, res_refac = arm_outputs["orig"], arm_outputs["refac"]
            eq_rep = reproduce_agent.verify_stagewise_equivalence(
                method_id=mid,
                slice_id=f"{dataset_tag}::{sample_tag}",
                X_prep_upstream=res_orig["X_preprocessed"],
                X_prep_modular=res_refac["X_preprocessed"],
                adj_upstream=res_orig["adj_used"],
                adj_modular=res_refac["adj_used"],
                emb_upstream=res_orig["embedding"],
                emb_modular=res_refac["embedding"],
                labels_upstream=native_labels if native_labels is not None else res_orig["pred_labels"],
                labels_modular=res_refac["pred_labels"],
                fallback_notes=res_refac["fallback_notes"],
            )
            stagewise_audit_logs.append(eq_rep.to_dict())

            if native_labels is not None:
                nat_agree = round(float(adjusted_rand_score(native_labels, res_refac["pred_labels"])), 4)
                native_agreement_by_method[mid] = nat_agree
            else:
                nat_agree = native_agreement_by_method.get(mid)

            # Determine explicit upstream equivalence gate status
            if res_refac["fallback_notes"]:
                gate_status = "FALLBACK_MODIFIED_PIPELINE"
            elif nat_agree is not None and nat_agree >= 0.85:
                gate_status = "VERIFIED_AGAINST_NATIVE_SCRIPT"
            elif nat_agree is not None and nat_agree < 0.85:
                gate_status = "DIVERGENT_FROM_NATIVE_SCRIPT"
            elif mid == "scanpy_nonspatial":
                gate_status = "BUILTIN_NONSPATIAL_BASELINE"
            else:
                gate_status = "INTERNAL_MODULAR_ONLY_NATIVE_NOT_RUN"

            if args.strict_native_gate and gate_status in ("FALLBACK_MODIFIED_PIPELINE", "DIVERGENT_FROM_NATIVE_SCRIPT"):
                print(f"   [GATE BLOCKED] Skipping `self`/`null` counterfactuals for {mid}: status={gate_status}")
                arms_to_log = ("orig", "refac")
            else:
                for cf_arm in ("self", "null"):
                    print(f"[Counterfactual Arm] {dataset_tag} (dose={dose}) | Method={mid} | Arm={cf_arm}")
                    arm_outputs[cf_arm] = run_chimera_counterfactual(
                        umi_csv=umi_csv,
                        coords_csv=coords_csv,
                        n_domains=n_dom,
                        preprocess_mod=mid,
                        adjacency_mod=mid,
                        train_mod=mid,
                        clust_mod=args.clust,
                        arm=cf_arm,
                        sedr_intervention_mode=args.sedr_mode,
                        null_seed=0,
                        output_dir=out_root / dataset_tag / f"dose_{dose:.2f}" / mid / cf_arm,
                    )
                arms_to_log = ("orig", "refac", "self", "null")  # type: ignore[assignment]

            for arm in arms_to_log:
                res = arm_outputs[arm]
                ari = float(adjusted_rand_score(y_true, res["pred_labels"]))
                records.append(
                    {
                        "dataset": dataset_tag,
                        "sample": sample_tag,
                        "dose": dose,
                        "requested_method": mid,
                        "effective_method_id": res["effective_method_id"],
                        "fallback_notes": res["fallback_notes"],
                        "sedr_intervention_mode": res["config"]["sedr_intervention_mode"],
                        "upstream_gate_status": gate_status,
                        "native_vs_refac_ari": nat_agree,
                        "graph_scale_mismatch_suspected": g_rep.scale_mismatch_suspected,
                        "arm": arm,
                        "clust": args.clust,
                        "ARI": round(ari, 4),
                    }
                )
                print(f"   -> [{arm}] effective={res['effective_method_id']} | gate={gate_status} | ARI={ari:.4f}")

    summary_path = out_root / "live_chimera_counterfactual_summary.json"
    summary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out_root / "native_reproduction_audit.json").write_text(json.dumps(native_audit_logs, indent=2), encoding="utf-8")
    (out_root / "stagewise_equivalence_audit.json").write_text(json.dumps(stagewise_audit_logs, indent=2), encoding="utf-8")
    (out_root / "graph_sanity_audit.json").write_text(json.dumps(graph_sanity_logs, indent=2), encoding="utf-8")
    print(f"\n[Completed] Saved audited counterfactual results & ReproduceAgent logs to: {out_root}")


if __name__ == "__main__":
    main()
