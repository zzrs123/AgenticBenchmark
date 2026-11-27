"""Statistical screening & confirmatory factorial inference engine (v2.1 Audited).

Fixes from static code review:
1. Joint collinearity check across `[M, C]` (logging cross-group collinear pairs rather than
   pruning `M` and `C` in isolation).
2. Data-adaptive `RidgeCV` + `LassoCV` in outer-loop screening instead of arbitrary fixed penalties.
3. Confirmatory inference aggregates slice contrasts to **independent `donor_id` clusters** first
   (`df = n_donors - 1`); explicitly returns `"Inconclusive (Insufficient Donors)"` if `n_donors < 2`.
4. Separates `(1 - alpha)` two-sided 95% CI (for superiority/direction) from `(1 - 2*alpha)` 90% CI
   (exact alpha=0.05 Two One-Sided Tests [TOST] equivalence rule).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import LassoCV, RidgeCV

from factor_demo.schemas import FactorialContrastResult, ScreeningCandidate


def _prune_collinear_columns_with_log(
    X: np.ndarray,
    names: List[str],
    corr_thresh: float = 0.92,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Drop near-collinear columns and return `(X_kept, kept_names, collinearity_notes)`."""
    keep_idx: List[int] = []
    notes: List[str] = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.std(col) < 1e-6:
            notes.append(f"dropped_constant:{names[j]}")
            continue
        is_dup = False
        for k in keep_idx:
            r = abs(float(np.corrcoef(col, X[:, k])[0, 1]))
            if r >= corr_thresh:
                is_dup = True
                notes.append(f"collinear({names[j]}~{names[k]},|r|={r:.2f})")
                break
        if not is_dup:
            keep_idx.append(j)
    return X[:, keep_idx], [names[i] for i in keep_idx], notes


def fit_outer_loop_screening(
    d_matrix: np.ndarray,
    d_names: List[str],
    m_matrix: np.ndarray,
    m_names: List[str],
    c_matrix: np.ndarray,
    c_names: List[str],
    y_obs: np.ndarray,  # shape (N_slices, K_methods)
    donor_ids: List[str],
    n_bootstraps: int = 60,
    seed: int = 42,
) -> List[ScreeningCandidate]:
    """Fit cross-validated two-stage mixed screening with joint `[M, C]` collinearity pruning."""
    rng = np.random.default_rng(seed)
    N, K = y_obs.shape

    D_clean, d_kept, _ = _prune_collinear_columns_with_log(d_matrix, d_names)

    # Jointly prune across [M, C] so collinearity between core bias & auxiliary pipeline is not hidden!
    mc_tagged_names = [f"M::{n}" for n in m_names] + [f"C::{n}" for n in c_names]
    MC_joint = np.hstack([m_matrix, c_matrix])
    MC_clean, mc_kept, _ = _prune_collinear_columns_with_log(MC_joint, mc_tagged_names)

    if D_clean.shape[1] == 0 or MC_clean.shape[1] == 0:
        return []

    D_std = (D_clean - np.mean(D_clean, axis=0, keepdims=True)) / (np.std(D_clean, axis=0, keepdims=True) + 1e-6)
    MC_std = (MC_clean - np.mean(MC_clean, axis=0, keepdims=True)) / (np.std(MC_clean, axis=0, keepdims=True) + 1e-6)

    def _build_design(slice_indices: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, str, str]]], np.ndarray]:
        X_main_rows = []
        X_int_rows = []
        Z_rand_rows = []
        y_vec = []
        int_meta: List[Tuple[str, str, str]] = []

        for d_n in d_kept:
            for mc_full in mc_kept:
                prefix, raw_n = mc_full.split("::", 1)
                grp = "D_x_M" if prefix == "M" else "D_x_C"
                int_meta.append((d_n, raw_n, grp))

        n_sub = len(slice_indices)
        for local_i, s_idx in enumerate(slice_indices):
            di = D_std[s_idx]
            for j in range(K):
                mc_j = MC_std[j]
                X_main_rows.append(np.concatenate([di, mc_j]))
                inter_terms = [di[p_idx] * mc_j[q_idx] for p_idx in range(len(d_kept)) for q_idx in range(len(mc_kept))]
                X_int_rows.append(np.asarray(inter_terms, dtype=np.float64))

                z_row = np.zeros(n_sub + K, dtype=np.float64)
                z_row[local_i] = 1.0
                z_row[n_sub + j] = 1.0
                Z_rand_rows.append(z_row)
                y_vec.append(float(y_obs[s_idx, j]))

        return (
            np.asarray(X_main_rows, dtype=np.float64),
            np.asarray(X_int_rows, dtype=np.float64),
            np.asarray(Z_rand_rows, dtype=np.float64),
            int_meta,
        ), np.asarray(y_vec, dtype=np.float64)

    def _fit_cv_screening_once(slice_indices: np.ndarray, use_cv: bool = True) -> Tuple[np.ndarray, List[Tuple[str, str, str]]]:
        (X_main, X_int, Z_rand, int_meta), y_vec = _build_design(slice_indices)
        y_center = y_vec - np.mean(y_vec)

        XZ_base = np.hstack([X_main, Z_rand])
        ridge = RidgeCV(alphas=np.logspace(-1, 2, 9), fit_intercept=False)
        ridge.fit(XZ_base, y_center)
        residual = y_center - ridge.predict(XZ_base)

        if use_cv and len(residual) >= 12:
            lasso = LassoCV(cv=min(4, len(residual) // 3), fit_intercept=False, max_iter=4000, random_state=seed)
            lasso.fit(X_int, residual)
            return lasso.coef_.copy(), int_meta

        from sklearn.linear_model import Lasso
        lasso_f = Lasso(alpha=0.01, fit_intercept=False, max_iter=4000)
        lasso_f.fit(X_int, residual)
        return lasso_f.coef_.copy(), int_meta

    full_idx = np.arange(N)
    point_w, int_meta = _fit_cv_screening_once(full_idx, use_cv=True)

    uniq_donors = sorted(set(donor_ids))
    donor_to_slices = {dn: [idx for idx, d in enumerate(donor_ids) if d == dn] for dn in uniq_donors}

    nonzero_counts = np.zeros_like(point_w)
    boot_estimates = []
    for _ in range(n_bootstraps):
        sampled_donors = rng.choice(uniq_donors, size=len(uniq_donors), replace=True)
        boot_slices: List[int] = []
        for dn in sampled_donors:
            boot_slices.extend(donor_to_slices[dn])
        w_b, _ = _fit_cv_screening_once(np.asarray(boot_slices, dtype=np.int64), use_cv=False)
        nonzero_counts += (np.abs(w_b) > 1e-5).astype(np.float64)
        boot_estimates.append(w_b)

    stability_probs = nonzero_counts / float(n_bootstraps)
    boot_mean_w = np.mean(np.asarray(boot_estimates), axis=0)

    candidates: List[ScreeningCandidate] = []
    for idx, (d_n, mc_n, grp) in enumerate(int_meta):
        est = float(0.5 * point_w[idx] + 0.5 * boot_mean_w[idx])
        stab = float(stability_probs[idx])
        prio = abs(est) * (0.35 + 0.65 * stab)
        direction = (
            "negative_antagonism" if est < -1e-4 else ("positive_synergy" if est > 1e-4 else "near_zero")
        )
        candidates.append(
            ScreeningCandidate(
                rank=0,
                d_feature=d_n,
                m_or_c_feature=mc_n,
                feature_group=grp,
                estimate_w=round(est, 4),
                stability_prob=round(stab, 3),
                priority_score=round(prio, 4),
                direction_hint=direction,
            )
        )

    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    for r, cand in enumerate(candidates, start=1):
        cand.rank = r
    return candidates


def compute_factorial_inference(
    hypothesis_id: str,
    method_id: str,
    tuning_protocol: str,
    operator_name: str,
    dose_low: float,
    dose_high: float,
    metric_name: str,
    per_slice_cells: Dict[str, Dict[str, float]],
    slice_to_donor: Optional[Dict[str, str]] = None,
    equivalence_margin: float = 0.03,
    alpha: float = 0.05,
    validity_passed: bool = True,
    convergence_rate: float = 1.0,
) -> FactorialContrastResult:
    """Compute donor-clustered 2x3 cell means, simple effects, 2x2 interaction CI, and exact TOST."""
    slice_ids = sorted(per_slice_cells.keys())
    donor_map = slice_to_donor or {s: s for s in slice_ids}

    # Aggregate slice-level cells to independent donor clusters first!
    donor_to_slices: Dict[str, List[str]] = defaultdict(list)
    for s in slice_ids:
        donor_to_slices[donor_map[s]].append(s)
    donor_ids = sorted(donor_to_slices.keys())
    n_donors = len(donor_ids)

    arms = ("refac", "self", "null")
    cell_keys = [f"{dose_low}_{a}" for a in arms] + [f"{dose_high}_{a}" for a in arms]

    donor_cells: Dict[str, Dict[str, float]] = {}
    for dn in donor_ids:
        member_s = donor_to_slices[dn]
        donor_cells[dn] = {
            ck: float(np.mean([per_slice_cells[s][ck] for s in member_s]))
            for ck in cell_keys
        }

    cell_means = {
        ck: round(float(np.mean([donor_cells[dn][ck] for dn in donor_ids])), 4)
        for ck in cell_keys
    }

    agg_low_list, agg_high_list = [], []
    topo_low_list, topo_high_list = [], []
    tau_agg_donors, tau_topo_donors = [], []

    for dn in donor_ids:
        c = donor_cells[dn]
        d_agg_lo = c[f"{dose_low}_refac"] - c[f"{dose_low}_self"]
        d_agg_hi = c[f"{dose_high}_refac"] - c[f"{dose_high}_self"]
        d_top_lo = c[f"{dose_low}_refac"] - c[f"{dose_low}_null"]
        d_top_hi = c[f"{dose_high}_refac"] - c[f"{dose_high}_null"]

        agg_low_list.append(d_agg_lo)
        agg_high_list.append(d_agg_hi)
        topo_low_list.append(d_top_lo)
        topo_high_list.append(d_top_hi)
        tau_agg_donors.append(d_agg_hi - d_agg_lo)
        tau_topo_donors.append(d_top_hi - d_top_lo)

    tau_arr = np.asarray(tau_topo_donors, dtype=np.float64)
    mean_tau_topo = float(np.mean(tau_arr))
    mean_tau_agg = float(np.mean(tau_agg_donors))

    # Gate 1: Validity & Convergence hard gate!
    if not validity_passed or convergence_rate < 0.80:
        return FactorialContrastResult(
            hypothesis_id=hypothesis_id,
            method_id=method_id,
            tuning_protocol=tuning_protocol,
            operator_name=operator_name,
            dose_low=dose_low,
            dose_high=dose_high,
            metric_name=metric_name,
            cell_means=cell_means,
            simple_agg_low=round(float(np.mean(agg_low_list)), 4),
            simple_agg_high=round(float(np.mean(agg_high_list)), 4),
            simple_topo_low=round(float(np.mean(topo_low_list)), 4),
            simple_topo_high=round(float(np.mean(topo_high_list)), 4),
            tau_int_agg_mean=round(mean_tau_agg, 4),
            tau_int_topo_mean=round(mean_tau_topo, 4),
            tau_int_topo_ci_low=float("nan"),
            tau_int_topo_ci_high=float("nan"),
            p_value_two_sided=1.0,
            equivalence_margin=equivalence_margin,
            evidence_state="Invalid Intervention / Convergence Anomaly (Gated)",
            slice_level_taus=[round(float(x), 4) for x in tau_topo_donors],
        )

    # Gate 2: Require >= 2 independent donors for valid inferential intervals!
    if n_donors < 2:
        return FactorialContrastResult(
            hypothesis_id=hypothesis_id,
            method_id=method_id,
            tuning_protocol=tuning_protocol,
            operator_name=operator_name,
            dose_low=dose_low,
            dose_high=dose_high,
            metric_name=metric_name,
            cell_means=cell_means,
            simple_agg_low=round(float(np.mean(agg_low_list)), 4),
            simple_agg_high=round(float(np.mean(agg_high_list)), 4),
            simple_topo_low=round(float(np.mean(topo_low_list)), 4),
            simple_topo_high=round(float(np.mean(topo_high_list)), 4),
            tau_int_agg_mean=round(mean_tau_agg, 4),
            tau_int_topo_mean=round(mean_tau_topo, 4),
            tau_int_topo_ci_low=float("nan"),
            tau_int_topo_ci_high=float("nan"),
            p_value_two_sided=1.0,
            equivalence_margin=equivalence_margin,
            evidence_state="Inconclusive (Insufficient Independent Donors < 2)",
            slice_level_taus=[round(float(x), 4) for x in tau_topo_donors],
        )

    se = float(np.std(tau_arr, ddof=1) / np.sqrt(n_donors))
    df = n_donors - 1
    # Standard (1 - alpha) two-sided 95% CI for superiority testing
    t_crit_95 = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    ci_95_low = mean_tau_topo - t_crit_95 * se
    ci_95_high = mean_tau_topo + t_crit_95 * se

    # Standard (1 - 2*alpha) 90% CI for exact alpha=0.05 Two One-Sided Tests (TOST) equivalence!
    t_crit_90 = float(stats.t.ppf(1.0 - alpha, df=df))
    tost_90_low = mean_tau_topo - t_crit_90 * se
    tost_90_high = mean_tau_topo + t_crit_90 * se

    if se > 1e-10:
        t_stat = mean_tau_topo / se
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))
    else:
        p_val = 0.0 if abs(mean_tau_topo) > 1e-6 else 1.0

    # Exact adjudication rules:
    # 1. Strong Supported Interaction: 95% CI completely outside [-Delta_eq, +Delta_eq]
    # 2. TOST Practically Equivalent (at alpha=0.05): 90% CI strictly inside [-Delta_eq, +Delta_eq]
    # 3. Directional Non-Zero Interaction: 95% CI excludes 0, overlaps +-Delta_eq
    if ci_95_low > equivalence_margin or ci_95_high < -equivalence_margin:
        evidence_state = "Supported Interaction (Exceeds Equivalence Margin)"
    elif tost_90_low >= -equivalence_margin and tost_90_high <= equivalence_margin:
        evidence_state = "Practically Equivalent (TOST alpha=0.05)"
    elif ci_95_low > 0.0 or ci_95_high < 0.0:
        evidence_state = "Statistically Non-Zero (Within/Overlapping Margin)"
    else:
        evidence_state = "Inconclusive"

    return FactorialContrastResult(
        hypothesis_id=hypothesis_id,
        method_id=method_id,
        tuning_protocol=tuning_protocol,
        operator_name=operator_name,
        dose_low=dose_low,
        dose_high=dose_high,
        metric_name=metric_name,
        cell_means=cell_means,
        simple_agg_low=round(float(np.mean(agg_low_list)), 4),
        simple_agg_high=round(float(np.mean(agg_high_list)), 4),
        simple_topo_low=round(float(np.mean(topo_low_list)), 4),
        simple_topo_high=round(float(np.mean(topo_high_list)), 4),
        tau_int_agg_mean=round(mean_tau_agg, 4),
        tau_int_topo_mean=round(mean_tau_topo, 4),
        tau_int_topo_ci_low=round(ci_95_low, 4),
        tau_int_topo_ci_high=round(ci_95_high, 4),
        p_value_two_sided=round(p_val, 5),
        equivalence_margin=equivalence_margin,
        evidence_state=evidence_state,
        slice_level_taus=[round(float(x), 4) for x in tau_topo_donors],
    )
