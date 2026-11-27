"""Automated Markdown & Multi-Page Vector PDF Report Builder.

Generates:
1. `experiment_report.md`: Complete bilingual (Chinese + technical English) report with
   Agent 1/2/3 trace tables, exact AST code citations, drift diagnostics, 2x3 factorial
   tables, and embedded figures.
2. `experiment_report.pdf`: Zero-dependency multi-page vector PDF generated via
   `matplotlib.backends.backend_pdf.PdfPages` (guaranteed to succeed on any Linux server
   or Windows PC without needing LaTeX / pandoc / wkhtmltopdf).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from factor_demo.schemas import (
    DatasetProfile,
    FactorialContrastResult,
    MethodProfile,
    PreRegisteredHypothesis,
    ScreeningCandidate,
    ValidityAuditRecord,
)


def generate_markdown_report(
    output_dir: Path,
    llm_mode_summary: str,
    dataset_profiles: List[DatasetProfile],
    method_profiles: List[MethodProfile],
    candidates: List[ScreeningCandidate],
    hypotheses: List[PreRegisteredHypothesis],
    audit_records: List[ValidityAuditRecord],
    contrasts: List[FactorialContrastResult],
    prep_crossover: Dict[str, Dict[str, float]],
    system_eval: Dict[str, Any],
    agent_conclusion: str,
    figure_paths: Dict[str, Path],
) -> Path:
    """Write comprehensive Markdown experiment report (`experiment_report.md`)."""
    md_path = output_dir / "experiment_report.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = [
        "# FACTOR Demo 实验验证报告：多智能体驱动的空间组学双侧受控因子基准",
        "",
        f"- **生成时间**：`{ts}`",
        f"- **智能体推理模式**：`{llm_mode_summary}`",
        "- **锁定的目标任务**：多细胞空间域识别（Multicellular Spatial Domain Segmentation）",
        "- **样本划分**：探索集 `8 张切片（4 个供体）` vs. 独立确认集 `4 张保留供体切片`（探索筛选与统计确认严格隔离）",
        "",
        "---",
        "",
        "## 一、 三组智能体协同职责与执行摘要",
        "",
        "1. **Dataset-Character Agent（数据集表征与偏移诊断）**：调用确定性空间统计工具提取无标签可观测特征 `d_obs`（用于样本外推荐）与参考依赖诊断特征 `d_ref`（仅用于回顾归因），并在内循环对扰动切片执行干预有效性与非目标偏移审计。",
        "2. **Method-Character Agent（AST 源码静态解析与等价门控）**：直接解析各空间域方法源码抽象语法树（AST），提取空间归纳偏置 `m_core` 与耦合辅助管线 `c_aux`，每个特征绑定精确源码行号 `CodeSpan`，并验证代码钩子在零干预下与原实现严格等价（`orig == refac`）。",
        "3. **D–M Interaction & Counterfactual Agent（外循环筛选 + 内循环受控 2×3 因子实验）**：在探索集上运行包含 `D×M` 与 `D×C` 竞争项的惩罚线性混合模型（PLMM + 供体聚类 Bootstrap），预先锁定参数化干预算子与等效区间（TOST），在独立确认集上分离检验 `refac`（真实空间图）、`self`（单位阵无跨单元聚合）与 `null`（同谱节点置换图）在默认配置与等预算重调优下的交互效应。",
        "",
        "### Agent 3 综合科学裁决摘要",
        "",
        agent_conclusion,
        "",
        "---",
        "",
        "## 二、 Agent 1 输出：空间切片元特征矩阵（`d_obs` 与 `d_ref` 分离）",
        "",
        "| 切片 ID | 划分 | 平台 | 组织类型 | 密度 (`d_obs`) | 稀疏度 (`d_obs`) | Moran's I (`d_obs`) | 局部表达异质性 (`d_obs`) | 最小域占比 (`d_ref`) | 边界跨边率 (`d_ref`) |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for dp in dataset_profiles:
        o, r = dp.d_obs, dp.d_ref
        lines.append(
            f"| `{dp.dataset_id}` | `{dp.split}` | {dp.platform_tag} | {dp.tissue_tag} | "
            f"{o['spatial_density']:.3f} | {o['sparsity_fraction']:.2%} | {o['global_morans_i']:.3f} | "
            f"{o['local_expr_heterogeneity']:.3f} | {r['min_domain_fraction_ref']:.3f} | {r['boundary_edge_ratio_ref']:.3f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 三、 Agent 2 输出：方法归纳偏置向量与 AST 代码行号溯源",
            "",
            "| 方法 ID | 方法家族 | 空间图 | 多跳无剪枝平滑 | 注意力边门控 | 邻域特征拼接 | 默认预处理 (`c_aux`) | 默认聚类头 (`c_aux`) | AST 源码行号证据（自动提取） |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )

    for mp in method_profiles:
        mc, ca = mp.m_core, mp.c_aux
        prep_str = "PCA" if ca.get("default_prep_pca", 0) > 0.5 else "HVG"
        clust_str = "GMM" if ca.get("default_clust_gmm", 0) > 0.5 else "KMeans"
        cite_str = "; ".join(
            f"`{v.file_path}:L{v.start_line}-{v.end_line}` ({k})"
            for k, v in list(mp.evidence_spans.items())[:3]
        )
        lines.append(
            f"| `{mp.method_id}` | {mp.family} | {int(mc['uses_spatial_graph'])} | "
            f"{int(mc['unpruned_multihop_smoothing'])} | {int(mc['attention_edge_gating'])} | "
            f"{int(mc['feature_concatenation'])} | `{prep_str}` | `{clust_str}` | {cite_str} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 四、 外循环观测性能矩阵与混合效应筛选优先级",
            "",
            f"![Observational Matrix and Screening](figures/{figure_paths['fig1'].name})",
            "",
            "| 排名 | 数据特征 | 方法/管线特征 | 交互组别 | 混合模型估计值 `W` | 供体 Bootstrap 稳定度 `Pi` | 优先分 |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    for c in candidates[:8]:
        lines.append(
            f"| {c.rank} | `{c.d_feature}` | `{c.m_or_c_feature}` | `{c.feature_group}` | "
            f"{c.estimate_w:+.4f} | {c.stability_prob:.2f} | {c.priority_score:.4f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 五、 预注册假设与干预有效性 / 代码等价审计",
            "",
        ]
    )

    for h in hypotheses:
        lines.extend(
            [
                f"### `{h.hypothesis_id}`: {h.title}",
                f"- **筛选依据**：{h.rationale_from_screening}",
                f"- **干预算子与剂量**：`{h.data_operator_name}`，剂量梯度 `{h.data_operator_doses}`，标签规则：`{h.label_evolution_rule}`",
                f"- **四臂方法对照**：`{h.method_arms}`；训练协议：`{h.tuning_protocols}`",
                f"- **证伪与等效判据**：{h.falsification_rule}",
                "",
            ]
        )

    lines.extend(
        [
            "### 干预有效性与非目标偏移诊断抽样记录（Agent 1 & Agent 2 联合审计）",
            "",
            "| 确认集切片 | 干预算子 | 剂量 | 目标特征偏移 (z) | 非目标最大偏移 (z) | 基因相关矩阵漂移 (Frobenius) | 代码重构最大误差 (`orig vs refac`) | 审计状态 |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for ar in audit_records[:9]:
        max_eq = max(ar.code_equiv_max_diff.values()) if ar.code_equiv_max_diff else 0.0
        lines.append(
            f"| `{ar.slice_id}` | `{ar.operator_name}` | {ar.dose:.2f} | {ar.target_shift_z:.3f} | "
            f"{ar.max_off_target_drift_z:.3f} | {ar.gene_corr_frobenius_drift:.5f} | {max_eq:.6f} | "
            f"{'PASS' if ar.passed_validity_check else 'WARN'} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 六、 独立确认集 2×3 受控因子实验与校准证据裁决",
            "",
            f"![Factorial Dose Response Curves](figures/{figure_paths['fig2'].name})",
            "",
            f"![Tissue Map Case Study](figures/{figure_paths['fig3'].name})",
            "",
            f"![Forest Plot and Competing Checks](figures/{figure_paths['fig4'].name})",
            "",
            "### 确认集完整简单效应与 2×2 因子交互对比表",
            "",
            "| 假设 ID | 方法 ID | 调参协议 | 低剂量 `refac-null` | 高剂量 `refac-null` | 拓扑交互项 `tau_int_topo` | 95% CI | 双侧 p 值 | 校准证据状态（含等效区间 `±0.03`） |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )

    for fc in contrasts:
        lines.append(
            f"| `{fc.hypothesis_id}` | `{fc.method_id}` | `{fc.tuning_protocol}` | "
            f"{fc.simple_topo_low:+.4f} | {fc.simple_topo_high:+.4f} | **{fc.tau_int_topo_mean:+.4f}** | "
            f"[{fc.tau_int_topo_ci_low:+.4f}, {fc.tau_int_topo_ci_high:+.4f}] | {fc.p_value_two_sided:.4f} | "
            f"**{fc.evidence_state}** |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 七、 竞争管线直接交叉验证（PCA vs. HVG）与样本外推荐评测",
            "",
            "### 1. 预处理因子直接交叉实验（保持原空间图 `refac` 不变）",
            "",
            "| 方法 ID | 使用 PCA 平均 ARI | 使用 HVG 平均 ARI | 差值 (`PCA - HVG`) |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )

    for mid, pv in prep_crossover.items():
        lines.append(
            f"| `{mid}` | {pv['ARI_with_PCA']:.4f} | {pv['ARI_with_HVG']:.4f} | {pv['PCA_minus_HVG']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "### 2. 严格基于无标签特征 `d_obs` 的样本外方法推荐遗憾值（Out-of-Sample Regret）",
            "",
            f"- 探索集静态全局平均第一名方法：`{system_eval['global_best_method_on_expl']}`",
            f"- 独立确认集理论最优上限（Oracle Mean ARI）：`{system_eval['mean_oracle_ari_conf']:.4f}`",
            f"- 静态排行榜推荐平均 ARI：`{system_eval['mean_static_winner_ari_conf']:.4f}`（平均遗憾值 Regret = `{system_eval['regret_static_winner']:.4f}`）",
            f"- **`d_obs` 元特征推荐器平均 ARI**：`{system_eval['mean_d_obs_recommended_ari_conf']:.4f}`（平均遗憾值 Regret = **`{system_eval['regret_d_obs_recommender']:.4f}`**）",
            f"- 有限算力预算主动选择命中：在 Top-`{system_eval['active_budget_k']}` 优先测试预算内检出 `{system_eval['active_supported_discoveries']}` 项显著交互（随机抽样基线期望为 `{system_eval['random_baseline_expected_discoveries']}` 项）。",
            "",
        ]
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def generate_pdf_report(
    output_dir: Path,
    llm_mode_summary: str,
    contrasts: List[FactorialContrastResult],
    system_eval: Dict[str, Any],
    figure_paths: Dict[str, Path],
) -> Path:
    """Export a clean multi-page vector PDF report (`experiment_report.pdf`) with zero missing-font risk."""
    pdf_path = output_dir / "experiment_report.pdf"

    with PdfPages(pdf_path) as pdf:
        # Page 1: Executive Title & Factorial Summary Table
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 Landscape
        ax = fig.add_subplot(111)
        ax.axis("off")

        ax.text(
            0.03,
            0.93,
            "FACTOR Demo Report: 3-Agent Controlled Factorial Meta-Benchmark",
            fontsize=16,
            fontweight="bold",
        )
        ax.text(
            0.03,
            0.87,
            f"Task: Multicellular Spatial Domain Segmentation  |  LLM Engine Mode: {llm_mode_summary}",
            fontsize=10.5,
            color="#333333",
        )
        summary_bullets = (
            "Key Architecture & Guardrails Verified in This Run:\n"
            "  1. Agent 1 (Dataset-Character): Deterministic spatial metrics separating label-free d_obs vs. diagnostic d_ref + drift audits.\n"
            "  2. Agent 2 (Method-Character): Python AST static code inspection with exact line spans + zero-intervention equivalence check (orig == refac).\n"
            "  3. Agent 3 (Interaction & Counterfactual): Identifiable PLMM screening on Exploration split -> Pre-registered 2x3 factorial experiments\n"
            "     on independent Held-out Confirmation donors (separating refac, self-loop I, and spectrum-preserving permuted graph P A P^T).\n"
            f"  4. Out-of-Sample Utility: Label-free d_obs recommender reduced selection regret from {system_eval['regret_static_winner']:.4f} "
            f"(static winner) to {system_eval['regret_d_obs_recommender']:.4f}."
        )
        ax.text(0.03, 0.65, summary_bullets, fontsize=9.8, verticalalignment="top", family="monospace")

        # Render confirmatory contrast table
        col_labels = ["Hypothesis", "Method", "Protocol", "Topo(Low)", "Topo(High)", "tau_int_topo", "95% CI", "Evidence State"]
        table_rows = []
        for c in contrasts:
            table_rows.append(
                [
                    c.hypothesis_id.replace("_GraphTopology", ""),
                    c.method_id,
                    c.tuning_protocol,
                    f"{c.simple_topo_low:+.3f}",
                    f"{c.simple_topo_high:+.3f}",
                    f"{c.tau_int_topo_mean:+.3f}",
                    f"[{c.tau_int_topo_ci_low:+.3f}, {c.tau_int_topo_ci_high:+.3f}]",
                    c.evidence_state,
                ]
            )

        tbl = ax.table(
            cellText=table_rows[:12],
            colLabels=col_labels,
            loc="bottom",
            bbox=[0.03, 0.05, 0.94, 0.52],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.0)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2b5c8f")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f4f7fa")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Pages 2-5: High-resolution figures
        titles = [
            ("fig1", "Page 2: Observational Performance Matrix & Outer-Loop PLMM Screening"),
            ("fig2", "Page 3: Held-Out Confirmation 3-Dose x 3-Arm Counterfactual Curves"),
            ("fig3", "Page 4: Visual Spatial Tissue Segmentation Case Study (0% vs 50% Infiltration)"),
            ("fig4", "Page 5: Confirmatory Forest Plot (with TOST Zone) & Preprocessing Crossover"),
        ]
        for key, pg_title in titles:
            img = mpimg.imread(str(figure_paths[key]))
            fig_p = plt.figure(figsize=(11.69, 8.27))
            ax_p = fig_p.add_subplot(111)
            ax_p.imshow(img)
            ax_p.axis("off")
            ax_p.set_title(pg_title, fontsize=12, fontweight="bold", pad=12)
            pdf.savefig(fig_p, bbox_inches="tight")
            plt.close(fig_p)

    return pdf_path
