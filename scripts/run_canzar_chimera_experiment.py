#!/usr/bin/env python3
"""Server CLI to run live 4-arm counterfactuals on Canzar 2026 real/simulator slices & Chimera methods.

Examples (inside `Chimera` or `scanpy+pytorch` Conda environment on server):
1. Run 4-arm (`orig`, `refac`, `self`, `null`) counterfactual on the built-in real `osmFISH_Codeluppi` slice:
   python scripts/run_canzar_chimera_experiment.py --use-osmfish --methods GraphST SEDR scanpy

2. Generate official mouse-brain snRNA-seq semi-synthetic slices (`subset_anndata.h5ad.gz`) across
   Type II cellular infiltration doses and run 4-arm + preprocessing crossover:
   python scripts/run_canzar_chimera_experiment.py --simulate-infiltration --methods GraphST STAGATE SEDR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_demo.canzar_bridge.canzar_data_loader import (
    DEFAULT_FRAMEWORK_DIR,
    generate_canzar_semisynthetic_slice,
)
from factor_demo.canzar_bridge.chimera_counterfactual_driver import (
    run_chimera_counterfactual,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live Canzar 2026 Chimera 4-arm counterfactuals.")
    parser.add_argument("--use-osmfish", action="store_true", help="Use real osmFISH_Codeluppi slice in external repo.")
    parser.add_argument("--simulate-infiltration", action="store_true", help="Generate Type II infiltration doses from subset_anndata.h5ad.gz.")
    parser.add_argument("--doses", nargs="+", type=float, default=[0.0, 0.25, 0.50], help="Infiltration doses.")
    parser.add_argument("--methods", nargs="+", default=["GraphST", "SEDR", "scanpy"], help="Chimera modules to test.")
    parser.add_argument("--clust", choices=["mclust", "kmeans"], default="mclust")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "canzar_live_chimera"))
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    records = []

    slice_tasks = []
    if args.use_osmfish or not args.simulate_infiltration:
        base = DEFAULT_FRAMEWORK_DIR / "resources" / "osmFISH_Codeluppi" / "correctedsample"
        slice_tasks.append(("osmFISH_Codeluppi", 0.0, Path(f"{base}_UMImatrix.csv"), Path(f"{base}_coords.csv"), Path(f"{base}_labels.csv")))

    if args.simulate_infiltration:
        for dose in args.doses:
            prefix = out_root / "sim_slices" / f"stripes_type2_dose_{dose:.2f}"
            u_csv, c_csv, l_csv = generate_canzar_semisynthetic_slice(
                output_prefix=prefix,
                shape="stripes",
                perturbation="type2_infiltration",
                dose=dose,
                seed=1,
            )
            slice_tasks.append((f"SimMouseBrain_Type2", dose, u_csv, c_csv, l_csv))

    for dataset_tag, dose, umi_csv, coords_csv, labels_csv in slice_tasks:
        y_true = pd.read_csv(labels_csv, header=None).iloc[:, 0].astype(str).to_numpy()
        n_dom = len(np.unique(y_true))
        for mid in args.methods:
            for arm in ("orig", "refac", "self", "null"):
                print(f"[Running] {dataset_tag} (dose={dose}) | Method={mid} | Arm={arm}")
                res = run_chimera_counterfactual(
                    umi_csv=umi_csv,
                    coords_csv=coords_csv,
                    n_domains=n_dom,
                    preprocess_mod=mid,
                    adjacency_mod=mid,
                    train_mod=mid,
                    clust_mod=args.clust,
                    arm=arm,
                    null_seed=0,
                    output_dir=out_root / dataset_tag / f"dose_{dose:.2f}" / mid / arm,
                )
                ari = float(adjusted_rand_score(y_true, res["pred_labels"]))
                records.append(
                    {
                        "dataset": dataset_tag,
                        "dose": dose,
                        "method": mid,
                        "arm": arm,
                        "clust": args.clust,
                        "ARI": round(ari, 4),
                    }
                )
                print(f"   -> ARI = {ari:.4f}")

    summary_path = out_root / "live_chimera_counterfactual_summary.json"
    summary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\n[Completed] Saved live counterfactual results to: {summary_path}")


if __name__ == "__main__":
    main()
