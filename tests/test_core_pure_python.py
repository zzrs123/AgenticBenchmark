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

    def test_v22_and_v23_audit_fixes_and_pure_python_behaviors(self) -> None:
        """Verify v2.2 & v2.3 wiring, behavioral unknown-mask screening, and bug fixes."""
        from scripts.run_canzar2026_real_audit import screen_real_interactions

        # 1. Behavioral pure-Python test: `screen_real_interactions` MUST mask `None` (`unknown`)
        #    rather than coercing `None -> 0.0`!
        expl_samples = ["S1", "S2", "S3", "S4"]
        eval_methods = ["M1", "M2", "M3", "M4", "M5", "M6"]
        d_features = {
            "S1": {"het": 0.10},
            "S2": {"het": 0.20},
            "S3": {"het": 0.40},
            "S4": {"het": 0.60},
        }
        # M5 and M6 have `None` ("unknown") for `feat_a`; if `None` were turned into 0.0, `n_known_methods` would be 6!
        m_profiles = {
            "M1": {"m_core_tristate": {"feat_a": 1.0}, "c_aux_tristate": {"prep_x": 0.0}},
            "M2": {"m_core_tristate": {"feat_a": 1.0}, "c_aux_tristate": {"prep_x": 1.0}},
            "M3": {"m_core_tristate": {"feat_a": 0.0}, "c_aux_tristate": {"prep_x": 0.0}},
            "M4": {"m_core_tristate": {"feat_a": 0.0}, "c_aux_tristate": {"prep_x": 1.0}},
            "M5": {"m_core_tristate": {"feat_a": None}, "c_aux_tristate": {"prep_x": 0.0}},
            "M6": {"m_core_tristate": {"feat_a": None}, "c_aux_tristate": {"prep_x": 1.0}},
        }
        ari_lookup = {}
        for i_s, s in enumerate(expl_samples):
            for m in eval_methods:
                base = 0.70 - 0.12 * i_s if m in ("M1", "M2") else 0.55
                # Inject extreme outlier on M5/M6 ("unknown" methods) to prove they are excluded from `feat_a`!
                if m in ("M5", "M6"):
                    base = -9.0 * i_s
                ari_lookup[(s, m)] = base

        cands = screen_real_interactions(expl_samples, eval_methods, ari_lookup, d_features, m_profiles)
        by_feat = {c["mc_feature"]: c for c in cands}
        self.assertEqual(by_feat["feat_a"]["n_known_methods"], 4)
        self.assertEqual(by_feat["feat_a"]["n_unknown_methods"], 2)
        self.assertEqual(by_feat["feat_a"]["coverage_note"], "masked_available_case")
        self.assertEqual(by_feat["prep_x"]["n_known_methods"], 6)
        self.assertEqual(by_feat["prep_x"]["n_unknown_methods"], 0)

        # 2. Verify `canzar_ast_inspector` no longer exports zero-filled `m_core`/`c_aux` dicts
        inspected = inspect_all_canzar_methods(DEFAULT_FRAMEWORK_DIR)
        for rec in inspected:
            self.assertNotIn("m_core", rec)
            self.assertNotIn("c_aux", rec)
            self.assertIn("unknown_mask", rec)

        # 3. Verify NumPy boolean-copy bug fix (`np.ix_`), marker-free `d_obs`, donor-grouped split, and no silent `ext_test = conf`
        gen_src = (PROJECT_ROOT / "factor_demo" / "data_lab" / "generator_and_loader.py").read_text(encoding="utf-8")
        self.assertIn("log_rates[np.ix_(in_dom, m_indices)] = 1.45", gen_src)
        self.assertNotIn("log_rates[in_dom][:, m_indices]", gen_src)
        self.assertIn("def assign_donor_grouped_splits", gen_src)

        dm_src = (PROJECT_ROOT / "factor_demo" / "tools" / "dataset_metrics.py").read_text(encoding="utf-8")
        self.assertIn("counts = slc.get_model_input_counts()", dm_src)

        pipe_src = (PROJECT_ROOT / "factor_demo" / "pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("ext_test_slices = conf_slices", pipe_src)
        self.assertIn("SKIPPED_NO_UNTOUCHED_EXTERNAL_TEST_DONORS", pipe_src)
        self.assertIn("verify_stagewise_equivalence(", pipe_src)

        # 4. Verify live Chimera CLI wires `ReproduceAgent`, `effective_method_id`, `fallback_notes`, and `scanpy_nonspatial`
        cli_src = (PROJECT_ROOT / "scripts" / "run_canzar_chimera_experiment.py").read_text(encoding="utf-8")
        self.assertIn("run_native_upstream_script(", cli_src)
        self.assertIn("verify_stagewise_equivalence(", cli_src)
        self.assertIn('"effective_method_id": res["effective_method_id"]', cli_src)
        self.assertIn('"fallback_notes": res["fallback_notes"]', cli_src)
        self.assertIn('"scanpy_nonspatial"', cli_src)

        # 5. Verify unconfounded Chimera 2x2 ablation, isotropic `gmm_spherical`, and `audit_lambda_weight_control`
        deliv_src = (PROJECT_ROOT / "factor_demo" / "deliverables.py").read_text(encoding="utf-8")
        self.assertIn('cluster_head: str = "gmm_spherical"', deliv_src)
        self.assertIn("def audit_lambda_weight_control", deliv_src)
        self.assertNotIn("k_nbr = 8 if loc_het > self.het_median_calib else 10", deliv_src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
