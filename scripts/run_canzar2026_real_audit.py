#!/usr/bin/env python3
"""Run the 3-Agent Framework directly on the REAL March 2026 Datasets & 35 Method Scripts!

Executes in pure Python 3.11+ (zero heavy external dependencies needed for this audit stage)
directly over the cloned `external/spatial-domain-analysis-framework` repository:
1. Agent 1 (`Dataset-Character`): Audits raw `osmFISH_Codeluppi` UMI counts (4,839 cells x 33 genes)
   AND extracts strictly separated `d_obs` (label-free landmarking) vs `d_ref` (GT diagnostics)
   across 82 real tissue sections from 9 datasets across 7 technologies.
2. Agent 2 (`Method-Character`): Statically inspects ALL 35 real `.py`/`.R` method scripts +
   `chimera_run.py`, extracting structured `m_core`, `c_aux`, hardcoded-seed warnings, and exact
   file:line `CodeSpan` provenance.
3. Agent 3 (`D-M Interaction & Counterfactual`):
   - Splits the 82 real tissue sections into 56 Exploration slices vs. 26 Held-Out Confirmation slices
     from unseen organs/platforms (`10xVisium_Fu` Breast Cancer, `MERFISH_Moffitt` Hypothalamus,
     `StereoSeq_Chen`, `Slide-seqV2_Lan`, `STARmap_Wang`);
   - Screens de-trended `D x M` and `D x C` interactions on the 56 real exploration slices;
   - Evaluates controlled multi-dose factorial contrasts on the March 2026 semi-synthetic perturbation
     sweeps (`exp8` spatial resolution aggregation & `exp2` Type II cellular infiltration);
   - Tests strictly label-free `d_obs` method recommendation on the 26 unseen real confirmation slices!
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_demo.canzar_bridge.canzar_ast_inspector import inspect_all_canzar_methods
from factor_demo.canzar_bridge.canzar_data_loader import (
    DATASET_TECH_SPECS,
    DEFAULT_FRAMEWORK_DIR,
    inspect_raw_3csv_slice_pure_python,
    load_canzar_real_benchmark_records,
)
from factor_demo.llm_client import DualModeLLMClient

PAPER_26_METHODS = {
    "ADEPT", "BANKSY", "BASS", "BayesSpace", "CCST", "CellCharter", "DeepST",
    "GraphPCA", "GraphST", "MERINGUE", "MNMST", "PAST", "PRECAST", "SC-MEB",
    "SCAN-IT", "SEDR", "SpaDo", "SpaGCN", "SpaceFlow", "SpatialMGCN",
    "SpatialPCA", "SpiceMix", "STAGATE", "TACCO", "UTAG", "Vesalius",
}
EXPLORATION_DATASETS = {
    "10xVisium_Maynard",
    "MERFISH_Zhang",
    "Slide-seq_Langlieb",
    "osmFISH_Codeluppi",
}
CONFIRMATION_DATASETS = {
    "10xVisium_Fu",
    "MERFISH_Moffitt",
    "STARmap_Wang",
    "Slide-seqV2_Lan",
    "StereoSeq_Chen",
}


def _zscore_list(vals: List[float]) -> List[float]:
    if len(vals) < 2:
        return [0.0 for _ in vals]
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
    if sd < 1e-8:
        return [0.0 for _ in vals]
    return [(v - mu) / sd for v in vals]


def screen_real_interactions(
    expl_samples: List[str],
    eval_methods: List[str],
    ari_lookup: Dict[Tuple[str, str], float],
    d_features: Dict[str, Dict[str, float]],
    m_profiles: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """De-trend sample & method main effects on 56 real exploration slices and screen D x (M, C)."""
    # Compute grand, sample, and method means on observed entries
    all_vals = [
        ari_lookup[(s, m)]
        for s in expl_samples
        for m in eval_methods
        if (s, m) in ari_lookup and math.isfinite(ari_lookup[(s, m)])
    ]
    grand_mu = sum(all_vals) / max(1, len(all_vals))

    s_means = {}
    for s in expl_samples:
        vs = [ari_lookup[(s, m)] for m in eval_methods if (s, m) in ari_lookup and math.isfinite(ari_lookup[(s, m)])]
        s_means[s] = (sum(vs) / len(vs) - grand_mu) if vs else 0.0

    m_means = {}
    for m in eval_methods:
        vm = [ari_lookup[(s, m)] for s in expl_samples if (s, m) in ari_lookup and math.isfinite(ari_lookup[(s, m)])]
        m_means[m] = (sum(vm) / len(vm) - grand_mu) if vm else 0.0

    # Residual matrix R_{i,j} = Y_{i,j} - mu - u_i - v_j
    residuals: Dict[Tuple[str, str], float] = {}
    for s in expl_samples:
        for m in eval_methods:
            if (s, m) in ari_lookup and math.isfinite(ari_lookup[(s, m)]):
                residuals[(s, m)] = ari_lookup[(s, m)] - grand_mu - s_means[s] - m_means[m]

    d_keys = sorted(next(iter(d_features.values())).keys())
    m_keys = sorted(next(iter(m_profiles.values()))["m_core_tristate"].keys())
    c_keys = sorted(next(iter(m_profiles.values()))["c_aux_tristate"].keys())

    # Standardize D across expl_samples; for M/C, strictly mask out `None` (`unknown`) per attribute!
    d_std_cols = {
        dk: dict(zip(expl_samples, _zscore_list([d_features[s][dk] for s in expl_samples])))
        for dk in d_keys
    }
    mc_specs: List[Tuple[str, str, Dict[str, float], int, int]] = []
    for grp, dict_key, keys in (
        ("D_x_M", "m_core_tristate", m_keys),
        ("D_x_C", "c_aux_tristate", c_keys),
    ):
        for fk in keys:
            known_m = [m for m in eval_methods if m_profiles[m][dict_key][fk] is not None]
            n_unk = len(eval_methods) - len(known_m)
            if len(known_m) < 4:
                continue
            z_vals = _zscore_list([float(m_profiles[m][dict_key][fk]) for m in known_m])
            if all(abs(z) < 1e-8 for z in z_vals):
                continue
            mc_specs.append((grp, fk, dict(zip(known_m, z_vals)), len(known_m), n_unk))

    candidates = []
    for dk in d_keys:
        for grp, f_name, m_col_known, n_known, n_unk in mc_specs:
            x_prod = []
            y_res = []
            for s in expl_samples:
                for m, m_z in m_col_known.items():
                    if (s, m) in residuals:
                        x_prod.append(d_std_cols[dk][s] * m_z)
                        y_res.append(residuals[(s, m)])
            denom = sum(x * x for x in x_prod)
            if denom < 1e-6:
                continue
            w_est = sum(x * y for x, y in zip(x_prod, y_res)) / denom
            r_num = sum(x * y for x, y in zip(x_prod, y_res))
            r_den = math.sqrt(denom * max(1e-8, sum(y * y for y in y_res)))
            corr = r_num / r_den
            unk_frac = round(n_unk / max(1, len(eval_methods)), 3)
            candidates.append(
                {
                    "d_feature": dk,
                    "mc_feature": f_name,
                    "group": grp,
                    "n_known_methods": n_known,
                    "n_unknown_methods": n_unk,
                    "unknown_fraction": unk_frac,
                    "coverage_note": "masked_available_case" if n_unk > 0 else "complete_26_methods",
                    "w_interaction": round(w_est, 4),
                    "partial_r": round(corr, 4),
                    "abs_score": round(abs(corr), 4),
                }
            )

    candidates.sort(key=lambda c: c["abs_score"], reverse=True)
    return candidates


def analyze_canzar_controlled_sweeps(
    framework_dir: Path,
    target_methods: List[str],
    eq_margin: float = 0.03,
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute external-baseline-referenced 2x2 contrasts on Canzar 2026 controlled sweeps (`exp8` & `exp2`).

    Uses `scanpy` (external non-spatial pipeline on exact same perturbed slice) as an observational
    reference baseline, measuring how whole-pipeline advantage `Delta(dose) = ARI_method(dose) - ARI_scanpy(dose)`
    changes across controlled doses (NOT interpreted as within-method single-component isolation).
    """
    scores_path = framework_dir / "results" / "scores.csv"
    synth_ds = {"synth-blobs", "synth-circles", "synth-moons", "synth-varied"}

    # exp8: spatial resolution grid length (0.5 = single-cell high-res vs 5.0 = Visium-like spot)
    # exp2: Type II cellular infiltration replacement proportion (0.0 vs 0.5)
    exp8_vals: Dict[Tuple[str, float, str], float] = {}
    exp2_vals: Dict[Tuple[str, float, str], float] = {}

    with scores_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            if ds not in synth_ds:
                continue
            sp = row["sample"]
            mt = row["method"]
            if mt not in target_methods and mt != "scanpy":
                continue
            try:
                ari = float(row["ARI"])
                if not math.isfinite(ari):
                    continue
            except ValueError:
                continue

            # Parse exp8 grid length: e.g. `exp8-gridlength0.5-sd1` or `exp8-gridlength5.0-sd1`
            if sp.startswith("exp8-gridlength"):
                rest = sp.split("exp8-gridlength", 1)[1]
                dose_str, rep_id = rest.split("-", 1)
                dose = float(dose_str)
                if dose in (0.5, 5.0):
                    exp8_vals[(f"{ds}::{rep_id}", dose, mt)] = ari

            # Parse exp2 Type II infiltration: e.g. `exp2-label23-prop0-sd1-cellsd1` or `prop0.5`
            elif sp.startswith("exp2-label23-prop") and "cellsd" in sp:
                rest = sp.split("exp2-label23-prop", 1)[1]
                dose_str, rep_id = rest.split("-", 1)
                dose = float(dose_str)
                if dose in (0.0, 0.5):
                    exp2_vals[(f"{ds}::{rep_id}", dose, mt)] = ari

    def _summarize_sweep(
        store: Dict[Tuple[str, float, str], float],
        dose_lo: float,
        dose_hi: float,
    ) -> List[Dict[str, Any]]:
        units = sorted({u for (u, _, _) in store.keys()})
        out = []
        for mt in target_methods:
            if mt == "scanpy":
                continue
            deltas_lo, deltas_hi, taus = [], [], []
            raw_lo, raw_hi = [], []
            for u in units:
                keys_needed = [(u, dose_lo, mt), (u, dose_lo, "scanpy"), (u, dose_hi, mt), (u, dose_hi, "scanpy")]
                if all(k in store for k in keys_needed):
                    d_lo = store[(u, dose_lo, mt)] - store[(u, dose_lo, "scanpy")]
                    d_hi = store[(u, dose_hi, mt)] - store[(u, dose_hi, "scanpy")]
                    raw_lo.append(store[(u, dose_lo, mt)])
                    raw_hi.append(store[(u, dose_hi, mt)])
                    deltas_lo.append(d_lo)
                    deltas_hi.append(d_hi)
                    taus.append(d_hi - d_lo)
            if len(taus) < 3:
                continue
            mean_tau = statistics.mean(taus)
            se = statistics.stdev(taus) / math.sqrt(len(taus))
            t_mult = 2.78 if len(taus) == 4 else 2.13
            ci_lo = mean_tau - t_mult * se
            ci_hi = mean_tau + t_mult * se
            if ci_lo > eq_margin or ci_high_check(ci_hi, -eq_margin):
                state = "Supported Interaction"
            elif ci_lo >= -eq_margin and ci_hi <= eq_margin:
                state = "Practically Equivalent"
            else:
                state = "Inconclusive"
            out.append(
                {
                    "method": mt,
                    "n_replicates": len(taus),
                    "ari_low": round(statistics.mean(raw_lo), 4),
                    "ari_high": round(statistics.mean(raw_hi), 4),
                    "pipeline_adv_low": round(statistics.mean(deltas_lo), 4),
                    "pipeline_adv_high": round(statistics.mean(deltas_hi), 4),
                    "tau_int": round(mean_tau, 4),
                    "ci_low": round(ci_lo, 4),
                    "ci_high": round(ci_hi, 4),
                    "evidence_state": state,
                }
            )
        return out

    return {
        "exp8_resolution_coarsening_0.5_to_5.0": _summarize_sweep(exp8_vals, 0.5, 5.0),
        "exp2_type2_infiltration_0.0_to_0.5": _summarize_sweep(exp2_vals, 0.0, 0.5),
    }


def ci_high_check(ci_hi: float, neg_margin: float) -> bool:
    return ci_hi < neg_margin


def evaluate_label_free_recommender_pure_python(
    expl_samples: List[str],
    conf_samples: List[str],
    eval_methods: List[str],
    ari_lookup: Dict[Tuple[str, str], float],
    d_obs_map: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Recommend methods on the 26 unseen real confirmation slices using strictly label-free `d_obs`."""
    obs_keys = sorted(next(iter(d_obs_map.values())).keys())
    # Standardize `d_obs` using ONLY exploration sample statistics
    means = {k: statistics.mean(d_obs_map[s][k] for s in expl_samples) for k in obs_keys}
    stds = {k: max(1e-6, statistics.pstdev(d_obs_map[s][k] for s in expl_samples)) for k in obs_keys}

    def _vec(s: str) -> List[float]:
        return [(d_obs_map[s][k] - means[k]) / stds[k] for k in obs_keys]

    # Global static winner on exploration slices
    expl_method_means = {}
    for m in eval_methods:
        vals = [ari_lookup[(s, m)] for s in expl_samples if (s, m) in ari_lookup and math.isfinite(ari_lookup[(s, m)])]
        expl_method_means[m] = statistics.mean(vals) if vals else 0.0
    static_winner = max(expl_method_means.items(), key=lambda kv: kv[1])[0]

    # Distance-weighted k-NN meta-regressor in standardized `d_obs` space (k=8 exploration slices)
    expl_vecs = {s: _vec(s) for s in expl_samples}
    regrets_static, regrets_agent = [], []
    realized_static, realized_agent, oracle_list = [], [], []
    by_dataset_summary: Dict[str, List[Tuple[float, float, float, str]]] = defaultdict(list)

    for cs in conf_samples:
        avail_m = [m for m in eval_methods if (cs, m) in ari_lookup and math.isfinite(ari_lookup[(cs, m)])]
        if len(avail_m) < 5 or static_winner not in avail_m:
            continue
        cv = _vec(cs)
        dists = []
        for es in expl_samples:
            ev = expl_vecs[es]
            d2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(cv, ev)))
            dists.append((d2, es))
        dists.sort(key=lambda x: x[0])
        top_nbrs = dists[:10]

        pred_m_scores = {}
        for m in avail_m:
            w_sum, val_sum = 0.0, 0.0
            for d_val, es in top_nbrs:
                if (es, m) in ari_lookup and math.isfinite(ari_lookup[(es, m)]):
                    w = 1.0 / (d_val + 0.25)
                    w_sum += w
                    val_sum += w * ari_lookup[(es, m)]
            local_est = (val_sum / w_sum) if w_sum > 0 else expl_method_means.get(m, 0.0)
            # Empirical-Bayes shrinkage toward global exploration mean to prevent small-k variance overfitting
            pred_m_scores[m] = 0.55 * expl_method_means.get(m, 0.0) + 0.45 * local_est

        chosen_m = max(pred_m_scores.items(), key=lambda kv: kv[1])[0]
        oracle_ari = max(ari_lookup[(cs, m)] for m in avail_m)
        stat_ari = ari_lookup[(cs, static_winner)]
        agt_ari = ari_lookup[(cs, chosen_m)]

        oracle_list.append(oracle_ari)
        realized_static.append(stat_ari)
        realized_agent.append(agt_ari)
        regrets_static.append(oracle_ari - stat_ari)
        regrets_agent.append(oracle_ari - agt_ari)
        ds_name = cs.split("::")[0]
        by_dataset_summary[ds_name].append((oracle_ari, stat_ari, agt_ari, chosen_m))

    ds_breakdown = {}
    for ds_n, items in sorted(by_dataset_summary.items()):
        top_rec = statistics.mode([x[3] for x in items])
        ds_breakdown[ds_n] = {
            "n_slices": len(items),
            "oracle_mean_ari": round(statistics.mean(x[0] for x in items), 4),
            "static_winner_ari": round(statistics.mean(x[1] for x in items), 4),
            "d_obs_agent_ari": round(statistics.mean(x[2] for x in items), 4),
            "most_recommended_method": top_rec,
        }

    return {
        "n_evaluated_conf_slices": len(oracle_list),
        "static_exploration_winner": static_winner,
        "mean_oracle_ari": round(statistics.mean(oracle_list), 4),
        "mean_static_winner_ari": round(statistics.mean(realized_static), 4),
        "mean_d_obs_agent_ari": round(statistics.mean(realized_agent), 4),
        "mean_regret_static": round(statistics.mean(regrets_static), 4),
        "mean_regret_d_obs_agent": round(statistics.mean(regrets_agent), 4),
        "heldout_dataset_breakdown": ds_breakdown,
    }


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs" / "canzar2026_real_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = DualModeLLMClient(mode="auto")

    print("[1/4] Agent 1: Auditing raw `osmFISH_Codeluppi` slice & 82 real tissue sections from Canzar 2026...")
    osmfish_stats = inspect_raw_3csv_slice_pure_python(DEFAULT_FRAMEWORK_DIR)
    raw_rows, d_obs_map, d_ref_map = load_canzar_real_benchmark_records(DEFAULT_FRAMEWORK_DIR, exclude_st_stahl=True)

    print("[2/4] Agent 2: Parsing AST & code structure of ALL 35 real method scripts + `chimera_run.py`...")
    method_inspections = inspect_all_canzar_methods(DEFAULT_FRAMEWORK_DIR)
    m_profile_map = {m["method_id"]: m for m in method_inspections}

    ari_lookup: Dict[Tuple[str, str], float] = {}
    for r in raw_rows:
        uid = f"{r['dataset']}::{r['sample']}"
        ari_lookup[(uid, r["method"])] = r["ARI"]

    all_samples = sorted(d_obs_map.keys())
    expl_samples = [s for s in all_samples if s.split("::")[0] in EXPLORATION_DATASETS]
    conf_samples = [s for s in all_samples if s.split("::")[0] in CONFIRMATION_DATASETS]

    # Retain exact 26 methods from Descoeudres et al. (2026) Fig. 1a
    eval_methods = sorted(
        m["method_id"]
        for m in method_inspections
        if m["method_id"] in PAPER_26_METHODS
    )

    d_merged = {s: {**d_obs_map[s], **d_ref_map[s]} for s in all_samples}

    print(f"[3/4] Agent 3 Outer Loop: Screening interactions on {len(expl_samples)} real Exploration slices x {len(eval_methods)} real methods...")
    candidates = screen_real_interactions(expl_samples, eval_methods, ari_lookup, d_merged, m_profile_map)

    print("[4/4] Agent 3 Inner Loop & Out-of-Sample Check: Analyzing controlled perturbation sweeps + 26 unseen Confirmation slices...")
    representative_methods = [
        "BASS", "BANKSY", "SpaceFlow", "SCAN-IT", "STAGATE", "GraphST", "SEDR", "CCST", "SpatialPCA", "SpaGCN"
    ]
    sweep_contrasts = analyze_canzar_controlled_sweeps(DEFAULT_FRAMEWORK_DIR, representative_methods)
    rec_eval = evaluate_label_free_recommender_pure_python(
        expl_samples, conf_samples, eval_methods, ari_lookup, d_obs_map
    )

    # Build comprehensive Markdown report grounded 100% in Canzar et al. (March 2026) real data & scripts
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hardcoded_seed_methods = [
        m["method_id"]
        for m in method_inspections
        if (m["c_aux_tristate"]["has_hardcoded_seed"] or 0.0) > 0.5
    ]

    md_lines = [
        "# 基于 Descoeudres et al. (2026.03) 真实数据集与 35 个真实方法的三智能体实测报告",
        "",
        f"- **运行时间**：`{ts}`",
        "- **底层数据与方法底座**：直接对接 `external/spatial-domain-analysis-framework` 与 `external/spatial-domain-simulator`",
        f"- **真实组织切片总数**：`{len(all_samples)}` 张切片（跨 9 个真实数据集；严格切分为 **{len(expl_samples)} 张探索集切片** vs. **{len(conf_samples)} 张跨器官/跨平台独立确认集切片**）",
        f"- **静态代码解析方法总数**：`{len(method_inspections)}` 个真实空间域脚本（25 个 `.py` + 10 个 `.R` + `chimera_run.py` 六大模块化深度模型）",
        "",
        "---",
        "",
        "## 一、 Agent 1 实测输出：原始 `osmFISH_Codeluppi` 矩阵审计与 9 大真实数据集特征概览",
        "",
        f"- **仓库内置原始单细胞切片实测 (`osmFISH_Codeluppi/correctedsample`)**：`{osmfish_stats['n_cells']}` 个细胞 × `{osmfish_stats['n_genes']}` 个基因，`{osmfish_stats['n_domains']}` 个皮层空间域，实测矩阵零值稀疏度 `{osmfish_stats['sparsity_fraction']:.2%}`，每细胞平均检出 `{osmfish_stats['mean_genes_per_cell']}` 个基因，最小空间域占比 `{osmfish_stats['min_domain_fraction']:.2%}`。",
        "",
        "### 9 大真实空间转录组数据集元特征均值（严格隔离无标签 `d_obs` 与真值诊断 `d_ref`）",
        "",
        "| 数据集 | 划分角色 | 切片数 | 平台类型 | 器官/组织 | 单细胞指标 (`d_obs`) | `scanpy` 空间离散度 PAS (`d_obs`) | 真值域空间边界度 `gt_PAS` (`d_ref`) | 真值转录本可分性 `gt_SI_X` (`d_ref`) |",
        "| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for ds_name in sorted(set(s.split("::")[0] for s in all_samples)):
        sub_s = [s for s in all_samples if s.startswith(ds_name + "::")]
        role = "Exploration" if ds_name in EXPLORATION_DATASETS else "**Held-Out Confirmation**"
        spec = DATASET_TECH_SPECS[ds_name]
        mean_sc_pas = statistics.mean(d_obs_map[s]["scanpy_PAS_spatial"] for s in sub_s)
        mean_gt_pas = statistics.mean(d_ref_map[s]["gt_PAS_spatial"] for s in sub_s)
        mean_gt_six = statistics.mean(d_ref_map[s]["gt_SI_expr"] for s in sub_s)
        md_lines.append(
            f"| `{ds_name}` | {role} | {len(sub_s)} | {spec['tech']} | {spec['organ']} | "
            f"{spec['is_single_cell']:.1f} | {mean_sc_pas:.3f} | {mean_gt_pas:.3f} | {mean_gt_six:+.3f} |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 二、 Agent 2 实测输出：真实方法源码静态解析与隐藏随机种子审计",
            "",
            f"- **自动源码审计发现**：在 35 个真实方法脚本中，Agent 2 通过语法树自动捕获到 **{len(hardcoded_seed_methods)} 个脚本在内部硬编码了随机种子**（`{', '.join(hardcoded_seed_methods)}`），从代码层面直接印证了论文关于“必须置换细胞输入顺序才能暴露隐藏随机性”的发现。",
            "- **Chimera 六大可插拔深度模型精确钩子定位**：",
            "",
            "| 方法名 | 类别 | 脚本文件 | HMRF/Potts 先验 | DGI 对比 | HVG (`c_aux`) | PCA (`c_aux`) | `mclust` (`c_aux`) | 字面量硬编码种子 | 包内待核查数 (`unknown`) | 自动提取的真实源码证据链 (`file:L#`) |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )

    def _fmt_tri(val: Any) -> str:
        return "`?`" if val is None else str(int(val))

    showcase_methods = [
        "GraphST", "STAGATE", "SEDR", "CCST", "SCAN-IT", "SpaceFlow",
        "BANKSY", "CellCharter", "BASS", "BayesSpace", "SpatialPCA", "SpaGCN", "GraphPCA", "TACCO"
    ]
    for mid in showcase_methods:
        if mid not in m_profile_map:
            continue
        mp = m_profile_map[mid]
        mc, ca = mp["m_core_tristate"], mp["c_aux_tristate"]
        spans = mp["evidence_spans"]
        cites = "; ".join(f"`{v.file_path}:L{v.start_line}` ({k})" for k, v in list(spans.items())[:3])
        md_lines.append(
            f"| `{mid}` | {mp['family']} | `{mp['script_file']}` | {_fmt_tri(mc['uses_hmrf_potts_prior'])} | "
            f"{_fmt_tri(mc['uses_dgi_contrastive'])} | {_fmt_tri(ca['uses_hvg_selection'])} | {_fmt_tri(ca['uses_pca_reduction'])} | "
            f"{_fmt_tri(ca['uses_mclust_gmm'])} | {_fmt_tri(ca['has_hardcoded_seed'])} | {len(mp['unknown_fields'])} | {cites} |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 三、 Agent 3 外循环：56 张真实探索集切片上的去主效应交互筛选（`D×M` vs `D×C`，已剔除 `unknown` 缺失掩码）",
            "",
            "| 排名 | 切片特征 (`d_obs` / `d_ref`) | 方法/管线属性 | 组别 | 已解析方法数 (`known/26`) | 交互斜率 `W` | 偏相关 `r` |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    for idx, c in enumerate(candidates[:10], start=1):
        md_lines.append(
            f"| {idx} | `{c['d_feature']}` | `{c['mc_feature']}` | `{c['group']}` | `{c['n_known_methods']}/26` | "
            f"{c['w_interaction']:+.4f} | {c['partial_r']:+.4f} |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 四、 既有受控半合成序列再分析：相对外部非空间 `scanpy` 基线的完整管线优势变化",
            "",
            "> **严谨性说明**：本节基于 Canzar 仓库预计算的 `scores.csv` 计算 `[ARI_M - ARI_scanpy]_high - [ARI_M - ARI_scanpy]_low`。由于 `M` 与 `scanpy` 同时存在预处理、表征与聚类头差异，该量反映**完整方法管线相对非空间基线的优势如何随数据条件变化**（作为观测与再分析线索），**不解释为单一空间图组件的内部净贡献**；同一方法内部的四臂（`orig/refac/self/null`）组件实验由 `chimera_counterfactual_driver.py` 在服务器上独立执行。",
            "",
            "### 1. 算子 `exp8`：空间分辨率退化（单细胞 `grid=0.5` $\\rightarrow$ Visium 级 `grid=5.0`）下相对外部 `scanpy` 的完整管线优势变化",
            "",
            "| 方法 | 重复单元数 | 高分辨 ARI (`0.5`) | 低分辨 ARI (`5.0`) | 高分辨相对管线优势 (`M - scanpy`) | 低分辨相对管线优势 (`M - scanpy`) | 2×2 交互变化量 `tau_int` | 95% CI | 校准证据状态 (`±0.03`) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )

    for row in sweep_contrasts["exp8_resolution_coarsening_0.5_to_5.0"]:
        md_lines.append(
            f"| `{row['method']}` | {row['n_replicates']} | {row['ari_low']:.3f} | {row['ari_high']:.3f} | "
            f"{row['pipeline_adv_low']:+.3f} | {row['pipeline_adv_high']:+.3f} | **{row['tau_int']:+.3f}** | "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | **{row['evidence_state']}** |"
        )

    md_lines.extend(
        [
            "",
            "### 2. 算子 `exp2`：Type II 细胞浸润异质性（`0%` $\\rightarrow$ `50%` 异质细胞置换）下相对外部 `scanpy` 的完整管线优势变化",
            "",
            "| 方法 | 重复单元数 | 0% 浸润 ARI | 50% 浸润 ARI | 0% 相对管线优势 (`M - scanpy`) | 50% 相对管线优势 (`M - scanpy`) | 2×2 交互变化量 `tau_int` | 95% CI | 校准证据状态 (`±0.03`) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )

    for row in sweep_contrasts["exp2_type2_infiltration_0.0_to_0.5"]:
        md_lines.append(
            f"| `{row['method']}` | {row['n_replicates']} | {row['ari_low']:.3f} | {row['ari_high']:.3f} | "
            f"{row['pipeline_adv_low']:+.3f} | {row['pipeline_adv_high']:+.3f} | **{row['tau_int']:+.3f}** | "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | **{row['evidence_state']}** |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 五、 独立确认集（26 张跨器官/跨技术保留真实切片）无标签 `d_obs` 推荐验证",
            "",
            f"- **训练范围**：仅使用 4 个探索集（`10xVisium_Maynard`, `MERFISH_Zhang`, `Slide-seq_Langlieb`, `osmFISH_Codeluppi` 共 56 张切片）的无标签特征 `d_obs`（分辨率 + 面板规模 + 非空间 `scanpy` 探针的 `PAS/CHAOS/SI_X`）。",
            f"- **测试范围**：完全未参与训练的 **5 个新数据集共 {rec_eval['n_evaluated_conf_slices']} 张真实切片**（涵盖人类乳腺癌 `10xVisium_Fu`、小鼠下丘脑 `MERFISH_Moffitt`、`StereoSeq_Chen`、`Slide-seqV2_Lan`、`STARmap_Wang`）。",
            f"- **探索集单一静态冠军（`{rec_eval['static_exploration_winner']}`）在确认集平均 ARI**：`{rec_eval['mean_static_winner_ari']:.4f}`（平均遗憾 Regret = `{rec_eval['mean_regret_static']:.4f}`）",
            f"- **`d_obs` 智能体推荐器在确认集平均 ARI**：**`{rec_eval['mean_d_obs_agent_ari']:.4f}`**（平均遗憾 Regret = **`{rec_eval['mean_regret_d_obs_agent']:.4f}`**）",
            "",
            "| 保留确认集（未见真实数据集） | 切片数 | 理论最优 Oracle ARI | 静态冠军 ARI | `d_obs` 推荐器 ARI | 最常推荐方法 |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]
    )

    for ds_n, info in rec_eval["heldout_dataset_breakdown"].items():
        md_lines.append(
            f"| `{ds_n}` | {info['n_slices']} | {info['oracle_mean_ari']:.4f} | "
            f"{info['static_winner_ari']:.4f} | **{info['d_obs_agent_ari']:.4f}** | `{info['most_recommended_method']}` |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 六、 在服务器上继续执行 Chimera 四臂（`orig/refac/self/null`）新干预实验命令",
            "",
            "上述真实切片审计完成后，可在服务器的 `Chimera` Conda 环境（`external/spatial-domain-analysis-framework/workflow/envs/chimera.yml`）直接对 `osmFISH_Codeluppi` 及 `subset_anndata.h5ad.gz` 运行全新的同谱置换图 `null` ($\\mathbf{P}\\mathbf{A}\\mathbf{P}^\\top$) 与自环 `self` ($\\mathbf{I}$) 实验：",
            "",
            "```bash",
            "python scripts/run_canzar_chimera_experiment.py --use-osmfish --simulate-infiltration --methods GraphST STAGATE SEDR SpaceFlow scanpy_nonspatial",
            "```",
            "",
        ]
    )

    report_path = out_dir / "canzar2026_real_agent_report.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    json_bundle_path = out_dir / "canzar2026_real_audit_bundle.json"
    json_bundle_path.write_text(
        json.dumps(
            {
                "osmfish_raw_audit": osmfish_stats,
                "n_methods_inspected": len(method_inspections),
                "hardcoded_seed_methods": hardcoded_seed_methods,
                "top_screening_candidates": candidates[:15],
                "controlled_sweep_contrasts": sweep_contrasts,
                "heldout_recommendation_eval": rec_eval,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[Done] Real Canzar 2026 Audit Markdown Report written to: {report_path}")


if __name__ == "__main__":
    main()
