#!/usr/bin/env python3
"""Command-line entrypoint to run the 3-Agent Spatial Omics Counterfactual Demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_demo.pipeline import run_demo_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FACTOR 3-Agent Counterfactual Meta-Benchmark Demo and export Markdown + PDF reports."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "demo_config.json"),
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "live", "offline"],
        default=None,
        help="LLM execution mode ('auto' uses API key if present, otherwise deterministic tool-grounded trace).",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Optional API key override.")
    parser.add_argument("--base-url", type=str, default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--model", type=str, default=None, help="Optional LLM model name override.")
    parser.add_argument(
        "--h5ad-dir",
        type=str,
        default=None,
        help="Optional directory of real .h5ad spatial slices (if omitted, runs built-in 12-slice cohort).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write figures, JSON traces, Markdown report, and PDF report.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast demo mode (1pert x 1null x 2train seeds, finishes in ~15-30 seconds on CPU).",
    )
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    llm_cfg = cfg.get("llm", {})
    exp_cfg = cfg.get("experiment", {})

    run_demo_pipeline(
        output_dir=args.output_dir or (PROJECT_ROOT / exp_cfg.get("output_dir", "outputs/demo_run")),
        llm_mode=args.mode or llm_cfg.get("mode", "auto"),
        api_key=args.api_key,
        base_url=args.base_url or llm_cfg.get("base_url"),
        model_name=args.model or llm_cfg.get("model"),
        h5ad_dir=args.h5ad_dir,
        n_pert_seeds=1 if args.quick else int(exp_cfg.get("n_perturbation_seeds", 2)),
        n_null_seeds=1 if args.quick else int(exp_cfg.get("n_graph_null_seeds", 2)),
        n_train_seeds=2 if args.quick else int(exp_cfg.get("n_training_seeds", 3)),
        equivalence_margin=float(exp_cfg.get("equivalence_margin_ari", 0.03)),
        verbose=True,
    )


if __name__ == "__main__":
    main()
