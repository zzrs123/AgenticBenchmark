"""Pure-Python AST static code inspector for spatial domain algorithm repositories.

Used by Agent 2 (`Method-Character Agent`) to extract verifiable inductive-bias and
auxiliary-pipeline descriptors anchored to exact source file line numbers (`CodeSpan`),
eliminating ungrounded LLM hallucination about method internals.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from factor_demo.schemas import CodeSpan


def _extract_snippet(lines: List[str], lineno: int, end_lineno: Optional[int]) -> str:
    start = max(0, lineno - 1)
    end = min(len(lines), end_lineno or lineno)
    snippet_lines = [ln.rstrip() for ln in lines[start : min(start + 4, end)]]
    return " | ".join(s.strip() for s in snippet_lines if s.strip())


def inspect_method_source(file_path: str | Path) -> Dict[str, object]:
    """Parse Python AST of a method module and return structured features + line-level CodeSpans."""
    path = Path(file_path).resolve()
    source_text = path.read_text(encoding="utf-8")
    lines = source_text.splitlines()
    tree = ast.parse(source_text, filename=str(path))

    # Default structured descriptors
    m_core: Dict[str, float] = {
        "uses_spatial_graph": 0.0,
        "unpruned_multihop_smoothing": 0.0,
        "attention_edge_gating": 0.0,
        "feature_concatenation": 0.0,
        "contrastive_dgi_loss": 0.0,
        "spatial_smooth_reg": 0.0,
    }
    c_aux: Dict[str, float] = {
        "default_prep_pca": 0.0,
        "default_prep_hvg": 0.0,
        "default_clust_gmm": 0.0,
        "default_clust_kmeans": 0.0,
    }
    spans: Dict[str, CodeSpan] = {}
    supported_hooks: List[str] = []
    module_docstring = ast.get_docstring(tree) or ""

    def _record_span(key: str, node: ast.AST, symbol: str) -> None:
        if key not in spans and hasattr(node, "lineno"):
            ln = int(getattr(node, "lineno"))
            end_ln = int(getattr(node, "end_lineno", ln))
            spans[key] = CodeSpan(
                file_path=path.name,
                start_line=ln,
                end_line=end_ln,
                symbol=symbol,
                snippet=_extract_snippet(lines, ln, end_ln),
            )

    for node in ast.walk(tree):
        # 1. Inspect runner function arguments for supported counterfactual hooks
        if isinstance(node, ast.FunctionDef) and node.name.startswith("run_"):
            arg_names = [a.arg for a in node.args.args]
            for candidate_hook in ("arm", "preprocess_override", "cluster_override", "tuning_protocol"):
                if candidate_hook in arg_names and candidate_hook not in supported_hooks:
                    supported_hooks.append(candidate_hook)
            _record_span("runner_entrypoint", node, node.name)

        # 2. Inspect function calls for spatial graph construction, concatenation, & hooks
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name == "build_spatial_knn_graph":
                m_core["uses_spatial_graph"] = 1.0
                _record_span("uses_spatial_graph", node, func_name)
            elif func_name == "_compute_attention_gated_graph":
                m_core["attention_edge_gating"] = 1.0
                _record_span("attention_edge_gating", node, func_name)
            elif func_name == "concatenate":
                m_core["feature_concatenation"] = 1.0
                _record_span("feature_concatenation", node, "np.concatenate")

        # 3. Inspect assignments for multi-hop message passing, DGI loss, spatial reg, & defaults
        if isinstance(node, ast.Assign):
            segment = ast.get_source_segment(source_text, node) or ""
            for target in node.targets:
                t_name = target.id if isinstance(target, ast.Name) else ""
                if t_name == "prop_adj" and "@" in segment:
                    m_core["unpruned_multihop_smoothing"] = 1.0
                    _record_span("unpruned_multihop_smoothing", node, "prop_adj = prop_adj @ adj")
                elif t_name == "bce_loss":
                    m_core["contrastive_dgi_loss"] = 1.0
                    _record_span("contrastive_dgi_loss", node, "DGI_bce_loss")
                elif t_name == "sp_loss":
                    m_core["spatial_smooth_reg"] = 1.0
                    _record_span("spatial_smooth_reg", node, "spatial_reg_loss")
                elif t_name == "prep_mode":
                    if '"pca"' in segment or "'pca'" in segment:
                        c_aux["default_prep_pca"] = 1.0
                        _record_span("default_prep", node, "prep_mode='pca'")
                    elif '"hvg"' in segment or "'hvg'" in segment:
                        c_aux["default_prep_hvg"] = 1.0
                        _record_span("default_prep", node, "prep_mode='hvg'")
                elif t_name == "clust_mode":
                    if '"gmm"' in segment or "'gmm'" in segment:
                        c_aux["default_clust_gmm"] = 1.0
                        _record_span("default_clust", node, "clust_mode='gmm'")
                    elif '"kmeans"' in segment or "'kmeans'" in segment:
                        c_aux["default_clust_kmeans"] = 1.0
                        _record_span("default_clust", node, "clust_mode='kmeans'")

    return {
        "file_name": path.name,
        "file_path": str(path),
        "docstring": module_docstring,
        "m_core": m_core,
        "c_aux": c_aux,
        "evidence_spans": spans,
        "supported_hooks": supported_hooks,
    }
