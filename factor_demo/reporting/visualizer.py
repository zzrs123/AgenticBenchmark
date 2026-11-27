"""Publication-style figure generator for the 3-Agent Counterfactual Meta-Benchmark Demo.

Uses clean English scientific labels inside matplotlib figures so rendering succeeds with
zero missing-glyph warnings on both Windows desktops and headless Linux bioinfo servers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from factor_demo.data_lab.generator_and_loader import SpatialSlice
from factor_demo.data_lab.perturbations import apply_data_operator
from factor_demo.methods_zoo.registry import METHODS_REGISTRY
from factor_demo.schemas import FactorialContrastResult, ScreeningCandidate
from factor_demo.tools.eval_metrics import align_clusters_to_reference


def plot_fig1_observational_and_screening(
    slice_ids: List[str],
    splits: List[str],
    method_ids: List[str],
    y_all: np.ndarray,
    candidates: List[ScreeningCandidate],
    out_path: Path,
) -> Path:
    """Figure 1: Observational ARI Heatmap (Expl + Conf) & Outer-Loop PLMM Screening Priority."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panel A: Slices x Methods ARI heatmap
    im = ax1.imshow(y_all, cmap="YlGnBu", aspect="auto", vmin=0.15, vmax=0.95)
    row_labels = [f"[{'E' if sp == 'exploration' else 'C'}] {sid}" for sid, sp in zip(slice_ids, splits)]
    ax1.set_yticks(np.arange(len(slice_ids)))
    ax1.set_yticklabels(row_labels, fontsize=8.5)
    ax1.set_xticks(np.arange(len(method_ids)))
    ax1.set_xticklabels(method_ids, rotation=22, ha="right", fontsize=9)
    ax1.axhline(sum(1 for sp in splits if sp == "exploration") - 0.5, color="#d62728", linestyle="--", linewidth=1.5)
    ax1.set_title("a  Observational Performance Matrix (ARI)\n(Dashed red line separates Exploration vs. Held-out Confirmation)", fontsize=10, fontweight="bold")

    for i in range(y_all.shape[0]):
        for j in range(y_all.shape[1]):
            val = y_all[i, j]
            txt_color = "white" if val > 0.62 else "black"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Mean ARI")

    # Panel B: Top-8 Outer-Loop PLMM Interaction Candidates
    top_c = candidates[:8][::-1]
    labels = [f"{c.d_feature}\nx {c.m_or_c_feature}" for c in top_c]
    scores = [c.priority_score for c in top_c]
    colors = ["#1f77b4" if c.feature_group == "D_x_M" else "#ff7f0e" for c in top_c]

    bars = ax2.barh(np.arange(len(top_c)), scores, color=colors, alpha=0.88, edgecolor="black", linewidth=0.6)
    ax2.set_yticks(np.arange(len(top_c)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Screening Priority Score (|W| * Stability Factor)", fontsize=9)
    ax2.set_title("b  Outer-Loop PLMM Screening (Blue: D x M, Orange: D x C)", fontsize=10, fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.5)

    for b_rect, c in zip(bars, top_c):
        ax2.text(
            b_rect.get_width() + 0.0008,
            b_rect.get_y() + b_rect.get_height() / 2.0,
            f"W={c.estimate_w:+.3f} (Pi={c.stability_prob:.2f})",
            va="center",
            fontsize=7.5,
        )
    ax2.set_xlim(0, max(scores) * 1.48 if scores else 0.1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig2_factorial_dose_curves(
    dose_curve_store: Dict[str, Any],
    hypothesis_id: str,
    method_ids: List[str],
    out_path: Path,
) -> Path:
    """Figure 2: 2x3 Counterfactual Dose-Response Curves across (`refac`, `self`, `null`) arms."""
    h_data = dose_curve_store[hypothesis_id]
    doses = h_data["doses"]
    fig, axes = plt.subplots(1, len(method_ids), figsize=(14.5, 3.9), sharey=True)
    if len(method_ids) == 1:
        axes = [axes]

    arm_styles = {
        "refac": ("Real Spatial Graph (refac)", "#1f77b4", "-o"),
        "self": ("Self-Loop Only (A = I)", "#2ca02c", "--s"),
        "null": ("Permuted Isomorphic Graph (P A P^T)", "#d62728", ":^"),
    }

    for idx, mid in enumerate(method_ids):
        ax = axes[idx]
        curves = h_data["methods"][mid]["fixed"]
        for arm, (lbl, col, fmt) in arm_styles.items():
            ax.plot(doses, curves[arm], fmt, color=col, linewidth=2.0, markersize=6, label=lbl)
        ax.set_title(mid, fontsize=10, fontweight="bold")
        ax.set_xlabel("Boundary Infiltration Dose (Swap Rate)", fontsize=8.5)
        if idx == 0:
            ax.set_ylabel("Held-Out Confirmation Mean ARI", fontsize=9)
        ax.set_xticks(doses)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_ylim(0.05, 0.95)

    axes[-1].legend(loc="lower left", fontsize=7.5, frameon=True)
    fig.suptitle(
        "Controlled 3-Dose x 3-Arm Factorial Curves on Held-Out Confirmation Donors (H1: Boundary Infiltration x Graph Topology)",
        fontsize=10.5,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig3_tissue_map_case_study(
    rep_slice: SpatialSlice,
    out_path: Path,
) -> Path:
    """Figure 3: Spatial Domain Map Comparison on a Representative Held-Out Slice at 0% vs 50% Infiltration."""
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6))
    cmap = "tab10"

    gcn_runner = METHODS_REGISTRY["GCN_DGI"]["runner"]  # type: ignore[index]
    banksy_runner = METHODS_REGISTRY["BANKSY_Concat"]["runner"]  # type: ignore[index]

    for row_idx, dose in enumerate([0.0, 0.50]):
        pert = apply_data_operator(rep_slice, "u2_boundary_infiltration", dose=dose, seed=0)
        y_true = pert.domain_labels
        x_in = pert.get_model_input_counts()

        gcn_refac = align_clusters_to_reference(
            y_true, gcn_runner(x_in, pert.coords, pert.n_domains, arm="refac", train_seed=0).pred_labels  # type: ignore[misc]
        )
        gcn_self = align_clusters_to_reference(
            y_true, gcn_runner(x_in, pert.coords, pert.n_domains, arm="self", train_seed=0).pred_labels  # type: ignore[misc]
        )
        banksy_refac = align_clusters_to_reference(
            y_true, banksy_runner(x_in, pert.coords, pert.n_domains, arm="refac", train_seed=0).pred_labels  # type: ignore[misc]
        )

        panels = [
            (y_true, f"Reference Domains\n(Infiltration={dose:.0%})"),
            (gcn_refac, f"GCN_DGI [refac: Real Graph]\n(Dose={dose:.0%})"),
            (gcn_self, f"GCN_DGI [self: A = I]\n(Dose={dose:.0%})"),
            (banksy_refac, f"BANKSY_Concat [refac]\n(Dose={dose:.0%})"),
        ]

        for col_idx, (labels_arr, title) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.scatter(
                pert.coords[:, 0],
                pert.coords[:, 1],
                c=labels_arr,
                cmap=cmap,
                s=13,
                alpha=0.9,
                vmin=0,
                vmax=9,
            )
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

    fig.suptitle(
        f"Visual Counterfactual Case Study on Held-Out Slice `{rep_slice.slice_id}`",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig4_forest_and_competing_checks(
    contrasts: List[FactorialContrastResult],
    prep_crossover: Dict[str, Dict[str, float]],
    system_eval: Dict[str, Any],
    out_path: Path,
) -> Path:
    """Figure 4: Forest Plot of 2x2 Interaction Contrasts (with TOST zone) & Competing Pipeline / Regret Checks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    h1_c = [c for c in contrasts if c.hypothesis_id == "H1_Infiltration_x_GraphTopology"]
    eq = h1_c[0].equivalence_margin if h1_c else 0.03

    ax1.axvspan(-eq, eq, color="#cccccc", alpha=0.45, label=f"Practical Equivalence Zone [-{eq:.2f}, +{eq:.2f}]")
    ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.0)

    y_pos = np.arange(len(h1_c))[::-1]
    labels = [f"{c.method_id} ({c.tuning_protocol})" for c in h1_c]
    for y, c in zip(y_pos, h1_c):
        col = "#d62728" if "Supported" in c.evidence_state else "#1f77b4"
        ax1.errorbar(
            c.tau_int_topo_mean,
            y,
            xerr=[[c.tau_int_topo_mean - c.tau_int_topo_ci_low], [c.tau_int_topo_ci_high - c.tau_int_topo_mean]],
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=2.0,
            capsize=4,
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_xlabel("2x2 Factorial Interaction Contrast tau_int_topo (95% CI across Donors)", fontsize=9)
    ax1.set_title("a  Confirmatory Interaction Contrasts (Fixed vs. Matched-Budget Retuned)", fontsize=9.8, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    # Panel B: Direct Preprocessing Crossover (PCA vs HVG) + Recommender Regret summary
    m_keys = list(prep_crossover.keys())
    x = np.arange(len(m_keys))
    width = 0.35
    pca_vals = [prep_crossover[m]["ARI_with_PCA"] for m in m_keys]
    hvg_vals = [prep_crossover[m]["ARI_with_HVG"] for m in m_keys]

    ax2.bar(x - width / 2, pca_vals, width, label="Preprocess: PCA", color="#4c72b0", edgecolor="black", linewidth=0.6)
    ax2.bar(x + width / 2, hvg_vals, width, label="Preprocess: HVG", color="#dd8452", edgecolor="black", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(m_keys, rotation=18, ha="right", fontsize=8.5)
    ax2.set_ylabel("Mean ARI on Confirmation Slices", fontsize=9)
    ax2.set_ylim(0.0, 0.95)
    ax2.set_title(
        f"b  Direct Preprocessing Crossover & Label-Free Recommender\n"
        f"(Out-of-sample Regret: Static={system_eval['regret_static_winner']:.3f} -> d_obs Agent={system_eval['regret_d_obs_recommender']:.3f})",
        fontsize=9.5,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path
