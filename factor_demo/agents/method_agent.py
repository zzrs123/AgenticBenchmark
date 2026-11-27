"""Agent 2: Method-Character Agent.

Responsibilities:
1. Run AST static source-code analysis (`inspect_method_source`) on each spatial domain method.
2. Cross-check code-level implementation against module documentation/paper descriptions,
   producing auditable `m_core` and `c_aux` descriptors backed by exact line spans (`CodeSpan`).
3. Execute the Code-Equivalence Control gate (`orig` vs. `refac`) before counterfactual runs.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from factor_demo.data_lab.generator_and_loader import SpatialSlice
from factor_demo.llm_client import DualModeLLMClient
from factor_demo.methods_zoo.registry import METHODS_REGISTRY
from factor_demo.schemas import MethodProfile
from factor_demo.tools.ast_code_inspector import inspect_method_source
from factor_demo.tools.eval_metrics import align_clusters_to_reference

SYSTEM_PROMPT = (
    "You are the Method-Character Agent in a spatial omics meta-benchmark. "
    "Given AST static inspection outputs and exact code spans, summarize the method's core "
    "spatial inductive bias (graph smoothing vs attention gating vs concatenation) and its "
    "default auxiliary pipeline (preprocessing & clustering head) in 2 concise sentences, "
    "explicitly citing line numbers."
)


class MethodCharacterAgent:
    """Extracts auditable method design vectors and validates hook equivalence."""

    def __init__(self, llm_client: DualModeLLMClient) -> None:
        self.llm = llm_client

    def profile_methods(self, method_ids: List[str] | None = None) -> List[MethodProfile]:
        """Parse ASTs and build verifiable MethodProfiles for registered methods."""
        ids = method_ids or list(METHODS_REGISTRY.keys())
        profiles: List[MethodProfile] = []

        for mid in ids:
            entry = METHODS_REGISTRY[mid]
            meta = entry["metadata"]  # type: ignore[index]
            src_path = str(entry["source_file"])
            trace = [f"call:inspect_method_source('{src_path}')"]
            ast_res = inspect_method_source(src_path)

            spans_dict = {
                k: {
                    "file": v.file_path,
                    "lines": f"L{v.start_line}-L{v.end_line}",
                    "symbol": v.symbol,
                }
                for k, v in ast_res["evidence_spans"].items()  # type: ignore[union-attr]
            }

            user_prompt = json.dumps(
                {
                    "method_id": mid,
                    "display_name": meta["display_name"],  # type: ignore[index]
                    "docstring": ast_res["docstring"],
                    "m_core": ast_res["m_core"],
                    "c_aux": ast_res["c_aux"],
                    "code_spans": spans_dict,
                },
                ensure_ascii=False,
            )

            def _offline_summary(
                m_id: str = mid,
                d_name: str = str(meta["display_name"]),  # type: ignore[index]
                mc: Dict[str, float] = ast_res["m_core"],  # type: ignore[assignment]
                ca: Dict[str, float] = ast_res["c_aux"],  # type: ignore[assignment]
                sp: Dict[str, Dict[str, str]] = spans_dict,
            ) -> str:
                active_biases = [k for k, v in mc.items() if v > 0.5]
                prep = "PCA" if ca.get("default_prep_pca", 0) > 0.5 else "HVG"
                clust = "GMM" if ca.get("default_clust_gmm", 0) > 0.5 else "KMeans"
                span_notes = ", ".join(f"{k}@{v['file']}:{v['lines']}" for k, v in list(sp.items())[:3])
                return (
                    f"{d_name} ({m_id}) implements spatial biases {active_biases} "
                    f"with default auxiliary pipeline [{prep} + {clust}]. "
                    f"Verified via AST spans: {span_notes}."
                )

            summary = self.llm.complete(
                agent_name="MethodCharacterAgent",
                stage=f"inspect_{mid}",
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                offline_fallback_fn=_offline_summary,
            )
            trace.append("verified_ast_code_spans_and_hooks")

            profiles.append(
                MethodProfile(
                    method_id=mid,
                    display_name=str(meta["display_name"]),  # type: ignore[index]
                    family=str(meta["family"]),  # type: ignore[index]
                    source_file=ast_res["file_name"],  # type: ignore[arg-type]
                    m_core=ast_res["m_core"],  # type: ignore[arg-type]
                    c_aux=ast_res["c_aux"],  # type: ignore[arg-type]
                    evidence_spans=ast_res["evidence_spans"],  # type: ignore[arg-type]
                    supported_hooks=ast_res["supported_hooks"],  # type: ignore[arg-type]
                    agent_summary=summary,
                    tool_trace=trace,
                )
            )
        return profiles

    def build_method_matrices(
        self,
        profiles: List[MethodProfile],
    ) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
        """Return (`M_matrix`, `m_names`, `C_matrix`, `c_names`)."""
        m_names = sorted(profiles[0].m_core.keys())
        c_names = sorted(profiles[0].c_aux.keys())
        M_mat = np.asarray([[p.m_core[k] for k in m_names] for p in profiles], dtype=np.float64)
        C_mat = np.asarray([[p.c_aux[k] for k in c_names] for p in profiles], dtype=np.float64)
        return M_mat, m_names, C_mat, c_names

    def verify_code_equivalence(
        self,
        slc: SpatialSlice,
        method_ids: List[str],
        seed: int = 0,
    ) -> Dict[str, float]:
        """Check `orig` vs. `refac` agreement (0.0 = exact match of embeddings & cluster assignments)."""
        discrepancies: Dict[str, float] = {}
        x_in = slc.get_model_input_counts()
        for mid in method_ids:
            runner = METHODS_REGISTRY[mid]["runner"]  # type: ignore[index]
            out_orig = runner(x_in, slc.coords, slc.n_domains, arm="orig", train_seed=seed)  # type: ignore[misc]
            out_refac = runner(x_in, slc.coords, slc.n_domains, arm="refac", train_seed=seed)  # type: ignore[misc]
            emb_diff = float(np.max(np.abs(out_orig.embedding - out_refac.embedding)))
            aligned = align_clusters_to_reference(out_orig.pred_labels, out_refac.pred_labels)
            label_disagree = float(np.mean(aligned != out_orig.pred_labels))
            discrepancies[mid] = round(max(emb_diff, label_disagree), 6)
        return discrepancies
