"""Zero-external-dependency verification tests runnable on bare Python 3.11+ (v2.2 Audited).

Verifies:
1. Syntax compilation (`py_compile`) of all `.py` modules across `factor_demo/` and `scripts/`.
2. `ast_code_inspector.py` extraction on self-contained `methods_zoo/` modules.
3. Tri-state `canzar_ast_inspector.py` on all 35 real scripts in `external/spatial-domain-analysis-framework`.
4. `DualModeLLMClient` offline grounded fallback.
5. Valid JSON & Python syntax across all cells in both Jupyter Notebooks.
6. Explicit source-level verification of all v2.1 & v2.2 audit fixes (preventing doc-code drift!).
"""

from __future__ import annotations

import ast
import json
import py_compile
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_demo.canzar_bridge.canzar_ast_inspector import inspect_all_canzar_methods
from factor_demo.canzar_bridge.canzar_data_loader import (
    DEFAULT_FRAMEWORK_DIR,
    inspect_raw_3csv_slice_pure_python,
)
from factor_demo.llm_client import DualModeLLMClient
from factor_demo.tools.ast_code_inspector import inspect_method_source


class TestFactorDemoIntegrity(unittest.TestCase):
    def test_all_python_files_compile_cleanly(self) -> None:
        py_files = list((PROJECT_ROOT / "factor_demo").rglob("*.py")) + list(
            (PROJECT_ROOT / "scripts").rglob("*.py")
        )
        self.assertGreaterEqual(len(py_files), 18)
        for pf in py_files:
            py_compile.compile(str(pf), doraise=True)

    def test_ast_inspector_extracts_accurate_method_profiles(self) -> None:
        zoo = PROJECT_ROOT / "factor_demo" / "methods_zoo"

        gcn = inspect_method_source(zoo / "gcn_dgi_domain.py")
        self.assertEqual(gcn["m_core"]["uses_spatial_graph"], 1.0)
        self.assertEqual(gcn["m_core"]["unpruned_multihop_smoothing"], 1.0)
        self.assertEqual(gcn["m_core"]["contrastive_dgi_loss"], 1.0)
        self.assertEqual(gcn["m_core"]["spatial_smooth_reg"], 1.0)
        self.assertEqual(gcn["c_aux"]["default_prep_hvg"], 1.0)
        self.assertEqual(gcn["c_aux"]["default_clust_gmm"], 1.0)
        self.assertIn("arm", gcn["supported_hooks"])

        gat = inspect_method_source(zoo / "gat_ae_domain.py")
        self.assertEqual(gat["m_core"]["attention_edge_gating"], 1.0)
        self.assertEqual(gat["c_aux"]["default_prep_pca"], 1.0)

        banksy = inspect_method_source(zoo / "banksy_aug_domain.py")
        self.assertEqual(banksy["m_core"]["feature_concatenation"], 1.0)
        self.assertEqual(banksy["c_aux"]["default_prep_pca"], 1.0)

    def test_canzar2026_real_repo_bridge_tristate(self) -> None:
        osm = inspect_raw_3csv_slice_pure_python(DEFAULT_FRAMEWORK_DIR)
        self.assertEqual(osm["n_cells"], 4839)
        self.assertEqual(osm["n_genes"], 33)
        self.assertEqual(osm["n_domains"], 11)

        inspected = inspect_all_canzar_methods(DEFAULT_FRAMEWORK_DIR)
        self.assertEqual(len(inspected), 35)
        by_id = {m["method_id"]: m for m in inspected}
        self.assertEqual(len(by_id["GraphST"]["chimera_hooks"]), 3)
        self.assertEqual(by_id["STAGATE"]["c_aux_tristate"]["has_hardcoded_seed"], 1.0)
        self.assertEqual(by_id["GraphPCA"]["m_core_tristate"]["uses_hmrf_potts_prior"], 0.0)
        self.assertEqual(by_id["SpatialPCA"]["m_core_tristate"]["uses_hmrf_potts_prior"], 0.0)
        self.assertEqual(by_id["BASS"]["m_core_tristate"]["uses_hmrf_potts_prior"], 1.0)
        self.assertEqual(by_id["BASS"]["c_aux_tristate"]["uses_mclust_gmm"], 0.0)
        self.assertEqual(by_id["STAGATE"]["c_aux_tristate"]["uses_mclust_gmm"], 1.0)
        self.assertIsNone(by_id["DeepST"]["c_aux_tristate"]["uses_hvg_selection"])

    def test_dual_mode_llm_offline_fallback(self) -> None:
        client = DualModeLLMClient(mode="offline")
        out = client.complete(
            agent_name="MethodCharacterAgent",
            stage="unit_test",
            system_prompt="sys",
            user_prompt="usr",
            offline_fallback_fn=lambda: "Grounded summary ok",
        )
        self.assertEqual(out, "Grounded summary ok")
        self.assertEqual(len(client.call_history), 1)

    def test_notebook_json_and_code_cells_parse(self) -> None:
        for nb_name in ("FACTOR_3Agent_Demo.ipynb", "FACTOR_Canzar2026_Real_Demo.ipynb"):
            nb_path = PROJECT_ROOT / "notebooks" / nb_name
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4)
            code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
            for idx, cell in enumerate(code_cells):
                ast.parse("".join(cell["source"]), filename=f"{nb_name}_cell_{idx}")

    def test_v22_audit_fixes_verified_in_source(self) -> None:
        """Prevent doc-code desync by verifying exact v2.1/v2.2 symbols & guards in source files."""
        # 1. ReproduceAgent native runner & edge-set Jaccard equivalence
        rep_src = (PROJECT_ROOT / "factor_demo" / "agents" / "reproduce_agent.py").read_text(encoding="utf-8")
        self.assertIn("def run_native_upstream_script", rep_src)
        self.assertIn("edge_set_jaccard", rep_src)
        self.assertIn("self_loop_match", rep_src)

        # 2. Deliverables: 4-way Chimera ablation, branch standardization, and low-rank SVD consensus (no dense N x N!)
        deliv_src = (PROJECT_ROOT / "factor_demo" / "deliverables.py").read_text(encoding="utf-8")
        self.assertIn('VARIANTS = ("fixed_concat", "gated_only", "adaptive_weight_only", "full_gated_adaptive")', deliv_src)
        self.assertIn("standardize_branch(", deliv_src)
        self.assertIn("preserve_relative_weights=True", deliv_src)
        self.assertIn("np.linalg.svd(B_tilde, full_matrices=False)", deliv_src)
        self.assertNotIn("one_hot @ one_hot.T", deliv_src)

        # 3. Base pipeline: global scalar scaling when preserve_relative_weights=True
        base_src = (PROJECT_ROOT / "factor_demo" / "methods_zoo" / "base_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("def standardize_branch", base_src)
        self.assertIn("preserve_relative_weights: bool = True", base_src)

        # 4. Chimera driver: SEDR encoder_only vs full_mechanism, pure nonspatial scanpy, fallback rename
        drv_src = (PROJECT_ROOT / "factor_demo" / "canzar_bridge" / "chimera_counterfactual_driver.py").read_text(encoding="utf-8")
        self.assertIn('sedr_intervention_mode: str = "encoder_only"', drv_src)
        self.assertIn("scanpy_nonspatial", drv_src)
        self.assertIn("pyGMM_fallback_for_R_mclust", drv_src)

        # 5. InteractionAgent: marker exclusion, pre_confirmation_queue, real convergence, crossed Dose x Prep
        int_src = (PROJECT_ROOT / "factor_demo" / "agents" / "interaction_agent.py").read_text(encoding="utf-8")
        self.assertIn("get_model_input_counts()", int_src)
        self.assertIn("self.pre_confirmation_queue", int_src)
        self.assertNotIn("convergence_ok_rate=1.0", int_src)
        self.assertIn("tau_int_prep_scheme", int_src)

        # 6. StatsEngine: RidgeCV/LassoCV, donor clustering, n_donors < 2 guard, 90% TOST CI
        stat_src = (PROJECT_ROOT / "factor_demo" / "tools" / "stats_engine.py").read_text(encoding="utf-8")
        self.assertIn("RidgeCV", stat_src)
        self.assertIn("LassoCV", stat_src)
        self.assertIn("if n_donors < 2:", stat_src)
        self.assertIn("t_crit_90", stat_src)

        # 7. Data Lab: 3-way split including external_test
        gen_src = (PROJECT_ROOT / "factor_demo" / "data_lab" / "generator_and_loader.py").read_text(encoding="utf-8")
        self.assertIn('"external_test"', gen_src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
