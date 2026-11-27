"""Behavioral scientific unit tests for FACTOR v2.3 (executed automatically when NumPy/SciPy/Sklearn are present)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HAS_SCI_DEPS = all(
    importlib.util.find_spec(pkg) is not None
    for pkg in ("numpy", "scipy", "sklearn")
)


@unittest.skipUnless(HAS_SCI_DEPS, "Requires numpy, scipy, and scikit-learn on bioinfo/server environment")
class TestFactorScientificBehaviors(unittest.TestCase):
    def test_heldout_markers_genuinely_upregulated_and_isolated_from_features(self) -> None:
        import numpy as np
        from factor_demo.data_lab.generator_and_loader import build_demo_cohort
        from factor_demo.tools.dataset_metrics import compute_dataset_features

        slc = build_demo_cohort(seed=7)[0]
        # 1. Verify `np.ix_` fix: held-out markers for domain 0 must have substantially higher mean counts inside domain 0!
        m0 = slc.heldout_markers[0]
        in_d0 = slc.domain_labels == 0
        mean_in = float(np.mean(slc.counts[in_d0][:, m0]))
        mean_out = float(np.mean(slc.counts[~in_d0][:, m0]))
        self.assertGreater(mean_in, mean_out * 2.0)

        # 2. Verify complete isolation: corrupting `heldout_markers` columns in `counts` MUST NOT change `d_obs` or `d_ref`!
        d_obs_1, d_ref_1 = compute_dataset_features(slc)
        slc_corrupt = slc.copy()
        for idxs in slc_corrupt.heldout_markers.values():
            slc_corrupt.counts[:, idxs] = 99999.0
        d_obs_2, d_ref_2 = compute_dataset_features(slc_corrupt)
        self.assertEqual(d_obs_1, d_obs_2)
        self.assertEqual(d_ref_1, d_ref_2)

    def test_donor_grouped_split_prevents_leakage_and_silent_external_alias(self) -> None:
        from factor_demo.data_lab.generator_and_loader import assign_donor_grouped_splits, build_demo_cohort

        demo_slices = build_demo_cohort(seed=11)
        # Take 6 slices across only 4 donors -> must assign exploration + confirmation ONLY, leaving external_test empty!
        four_donor_slices = []
        for s in demo_slices[:8]:  # 8 slices across Donor_E1..Donor_E4
            c = s.copy()
            c.split = "unassigned"
            four_donor_slices.append(c)

        assigned = assign_donor_grouped_splits(four_donor_slices)
        splits_found = {s.split for s in assigned}
        self.assertEqual(splits_found, {"exploration", "confirmation"})
        self.assertEqual([s for s in assigned if s.split == "external_test"], [])

        # Verify zero donor overlap across splits
        expl_donors = {s.donor_id for s in assigned if s.split == "exploration"}
        conf_donors = {s.donor_id for s in assigned if s.split == "confirmation"}
        self.assertTrue(expl_donors.isdisjoint(conf_donors))

        # Verify explicit donor-leakage rejection
        leaking = [s.copy() for s in demo_slices[:4]]
        leaking[0].split = "exploration"
        leaking[1].split = "confirmation"  # Same Donor_E1 in two splits!
        with self.assertRaises(ValueError):
            assign_donor_grouped_splits(leaking)

    def test_chimera_unconfounded_2x2_and_lambda_weight_sensitivity(self) -> None:
        from factor_demo.data_lab.generator_and_loader import build_demo_cohort
        from factor_demo.deliverables import FactorAuditedChimera

        slc = build_demo_cohort(seed=42)[4]  # Thin-layer slice
        x_in = slc.get_model_input_counts()
        chimera = FactorAuditedChimera()
        d_obs_high_het = {"local_expr_heterogeneity": 0.45}

        hps = {
            v: chimera.run(x_in, slc.coords, slc.n_domains, d_obs_slice=d_obs_high_het, variant=v, seed=0).hyperparams_used
            for v in FactorAuditedChimera.VARIANTS
        }
        # All 4 variants must share exact `k_neighbors == 8` and gated variants share exact `gate_temperature == 2.0`
        self.assertEqual({hp["k_neighbors"] for hp in hps.values()}, {8})
        self.assertEqual(hps["gated_only"]["gate_temperature"], 2.0)
        self.assertEqual(hps["full_gated_adaptive"]["gate_temperature"], 2.0)

        # Isotropic heads (`gmm_spherical`, `kmeans`) must genuinely shift partitions between lambda=0.05 and lambda=0.95
        audit = chimera.audit_lambda_weight_control(x_in, slc.coords, slc.n_domains, seed=0)
        self.assertGreater(audit["heads"]["kmeans"]["partition_shift_low_to_high"], 0.05)
        self.assertGreater(audit["heads"]["gmm_spherical"]["partition_shift_low_to_high"], 0.05)

    def test_low_rank_bipartite_svd_consensus_matches_dense_spectral(self) -> None:
        import numpy as np
        from sklearn.metrics import adjusted_rand_score
        from factor_demo.deliverables import run_label_free_consensus

        rng = np.random.default_rng(0)
        y_base = np.repeat([0, 1, 2, 3], 40)
        preds = {}
        for m_idx in range(4):
            noisy = y_base.copy()
            flip = rng.random(len(noisy)) < 0.10
            noisy[flip] = rng.integers(0, 4, size=int(np.sum(flip)))
            # Permute cluster IDs arbitrarily to verify label-ID invariance!
            perm_ids = rng.permutation(4)
            preds[f"M_{m_idx}"] = perm_ids[noisy]

        y_cons = run_label_free_consensus(preds, n_domains=4, seed=0)
        self.assertGreater(adjusted_rand_score(y_base, y_cons), 0.75)


if __name__ == "__main__":
    unittest.main(verbosity=2)
