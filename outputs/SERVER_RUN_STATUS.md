# Server Execution Status (results(v2.3-milestone1): server 10/10 behavioral tests and v2.3 demo figures/reports)

- **Timestamp**: `2026-11-27 23:38:11 CST`
- **Host**: `li.npuers.com`
- **Git Commit**: `301e317`

## Latest Log Tail (`logs_server/v23_exec.log`)
```text
[Start] Fri 27 Nov 2026 11:38:02 PM CST | Selected GPU: CUDA_VISIBLE_DEVICES=7
=== [Step 1] Running full 10-test suite (including 4 scientific behavioral tests) ===
test_all_python_files_compile_cleanly (test_core_pure_python.TestFactorDemoIntegrity) ... ok
test_ast_inspector_extracts_accurate_method_profiles (test_core_pure_python.TestFactorDemoIntegrity) ... ok
test_canzar2026_real_repo_bridge_tristate (test_core_pure_python.TestFactorDemoIntegrity) ... ok
test_dual_mode_llm_offline_fallback (test_core_pure_python.TestFactorDemoIntegrity) ... ok
test_notebook_json_and_code_cells_parse (test_core_pure_python.TestFactorDemoIntegrity) ... ok
test_v22_and_v23_audit_fixes_and_pure_python_behaviors (test_core_pure_python.TestFactorDemoIntegrity)
Verify v2.2 & v2.3 wiring, behavioral unknown-mask screening, and bug fixes. ... ok
test_chimera_unconfounded_2x2_and_lambda_weight_sensitivity (test_scientific_behaviors.TestFactorScientificBehaviors) ... ok
test_donor_grouped_split_prevents_leakage_and_silent_external_alias (test_scientific_behaviors.TestFactorScientificBehaviors) ... ok
test_heldout_markers_genuinely_upregulated_and_isolated_from_features (test_scientific_behaviors.TestFactorScientificBehaviors) ... ok
test_low_rank_bipartite_svd_consensus_matches_dense_spectral (test_scientific_behaviors.TestFactorScientificBehaviors) ... ok

----------------------------------------------------------------------
Ran 10 tests in 6.812s

OK
=== [Step 2] Reusing fresh v2.3 outputs/demo_run already computed on server ===
=== [Step 3] Running v2.3 Real Canzar 2026 Audit (unknown-masked AST + 46k scores) ===
[1/4] Agent 1: Auditing raw `osmFISH_Codeluppi` slice & 82 real tissue sections from Canzar 2026...
[2/4] Agent 2: Parsing AST & code structure of ALL 35 real method scripts + `chimera_run.py`...
[3/4] Agent 3 Outer Loop: Screening interactions on 56 real Exploration slices x 26 real methods...
[4/4] Agent 3 Inner Loop & Out-of-Sample Check: Analyzing controlled perturbation sweeps + 26 unseen Confirmation slices...

[Done] Real Canzar 2026 Audit Markdown Report written to: /home/dugaoyuan/AAA-new-base/hkust/AgenticBenchmark/outputs/canzar2026_real_audit/canzar2026_real_agent_report.md
```
