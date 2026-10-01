"""Phase 2 tests: metric correctness, leakage guards, reproducibility.  python -m unittest tests.test_phase2 -v
(needs artifacts/phase2/* from scripts/run_phase2.py for the output-consistency tests)"""
import json, re, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from kestrel import config as C, features as F, evaluate as E, models as M, pipeline as P, splits as S
from kestrel.clean import load_all

OUT = C.ART / "phase2"

class TestMetrics(unittest.TestCase):
    y = np.array([1, 0, 1, 0, 0, 0]); amt = np.array([1000., 500., 3000., 800., 200., 100.]); jit = np.zeros(6)
    s = np.array([.9, .8, .7, .1, .05, .01])

    def test_topk_and_rupees(self):
        t = E.topk_stats(self.y, self.amt, self.s, self.jit, 3)
        self.assertEqual((t["tp"], t["fp"]), (2, 1))
        self.assertAlmostEqual(t["precision"], 2 / 3); self.assertAlmostEqual(t["recall"], 1.0)
        self.assertAlmostEqual(t["stopped_inr"], 4000); self.assertAlmostEqual(t["stopped_per_check"], 4000 / 3)
        self.assertAlmostEqual(t["net_per_check"], (4000 - 380) / 3)
        self.assertAlmostEqual(t["net_per_check_after_contact"], (4000 - 380) / 3 - 260)

    def test_oracle_and_ev(self):
        self.assertAlmostEqual(E.oracle_per_check(self.y, self.amt, 1), 3000)
        self.assertAlmostEqual(float(E.expected_value(0.5, 1000.)), 0.5 * 1000 - 0.5 * 380)
        self.assertAlmostEqual(float(E.expected_value(380 / (1000 + 380), 1000.)), 0, places=6)   # break-even

    def test_threshold_rule(self):
        p = np.array([.9, .8, .7, .1, .05, .01]); r = E.threshold_stats(self.y, self.amt, p)
        self.assertEqual(r["n_flagged"], 3)           # claims 1-3 have EV>0; claim 4 (p=.1, Rs800) EV<0
        self.assertAlmostEqual(r["net_inr"], 4000 - 380)

    def test_all_negative_accuracy_is_not_a_result(self):
        self.assertAlmostEqual(1 - self.y.mean(), 4 / 6)

    def test_ties_broken_by_jitter_not_label(self):
        o = E.order(np.ones(5), np.array([.5, .1, .9, .3, .2])); self.assertEqual(list(o), [1, 4, 3, 0, 2])

    def test_bootstrap_reproducible(self):
        rng = np.random.default_rng(0); n = 300
        y = (rng.random(n) < .1).astype(int); amt = rng.uniform(300, 3000, n); part = rng.integers(0, 30, n); jit = rng.random(n)
        sc = {"rule_prior": rng.random(n), "m": y + rng.random(n)}
        a = E.cluster_bootstrap(y, amt, part, jit, 15, sc, reps=40, seed=1); b = E.cluster_bootstrap(y, amt, part, jit, 15, sc, reps=40, seed=1)
        pd.testing.assert_frame_equal(a, b)
        ci = E.summarise_ci(a, ["rule_prior", "m"], ["pr_auc"]); self.assertTrue((ci.lo <= ci.hi).all())

class TestLeakageGuards(unittest.TestCase):
    def test_phase2_never_touches_test_set(self):
        src = "".join((ROOT / f).read_text() for f in ("scripts/run_phase2.py", "kestrel/models.py", "kestrel/evaluate.py"))
        for bad in ("test_features", "features_test", "test_unlabelled", "test_clean", "load_all()[\"test\"]", "ds[\"test\"]"):
            self.assertNotIn(bad, src, bad)

    def test_no_random_split(self):
        src = "".join((ROOT / f).read_text() for f in ("scripts/run_phase2.py", "kestrel/models.py", "kestrel/evaluate.py"))
        self.assertIsNone(re.search(r"train_test_split|KFold|ShuffleSplit|cross_val", src))

    def test_every_candidate_feature_set_is_allowed(self):
        for kind in M.GRIDS: self.assertIn(kind, M.SIMPLICITY_ORDER)
        allowed = set(c for g in F.GROUPS.values() for c in g)
        self.assertFalse(allowed & F.FORBIDDEN)

@unittest.skipUnless((OUT / "model_selection.json").exists(), "run scripts/run_phase2.py first")
class TestOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sel = json.load(open(OUT / "model_selection.json")); cls.cmp = pd.read_csv(OUT / "model_comparison_by_fold.csv")

    def test_selection_record(self):
        self.assertEqual(self.sel["selected_hyperparams_on"], "A_pre_regime only")
        self.assertFalse(set(self.sel["final_features"]) & F.FORBIDDEN)
        self.assertIn(self.sel["chosen_family"], ["lr", "hgb", "rf", "RULE"])

    def test_all_models_all_folds_present(self):
        exp = {"rule_prior", "rule_partner", "lr", "hgb", "rf"}
        for f in ("A_pre_regime", "B_test_like", "C_june"):
            self.assertEqual(set(self.cmp[self.cmp.fold == f].model), exp)

    def test_accuracy_baseline_and_capacity(self):
        self.assertTrue(np.allclose(self.cmp.accuracy_all_negative, 1 - self.cmp.prevalence))
        self.assertEqual(sorted(self.cmp.groupby("fold").k_tp.size().index), ["A_pre_regime", "B_test_like", "C_june"])
        self.assertTrue((self.cmp.k_precision.between(0, 1)).all() and (self.cmp.pr_auc.between(0, 1)).all())

    def test_bootstrap_files_have_intervals(self):
        b = pd.read_csv(OUT / "bootstrap_ci.csv"); self.assertTrue((b.lo <= b.hi).all())
        self.assertEqual(set(b.fold), {"A_pre_regime", "B_test_like", "C_june"})

    def test_reproduce_one_fit(self):          # LR on fold C must reproduce the recorded PR-AUC
        ds = load_all(); X = P.train_features(ds); fold = S.FOLDS[2]
        tr, va = P.fold_data(ds, fold, X)
        p, _ = M.fit_predict("lr", next(c for c in M.GRIDS["lr"] if c["name"] == self.sel["configs"]["lr"]["name"]), F.feature_names(), tr, va, seed=0)
        rec = float(self.cmp[(self.cmp.model == "lr") & (self.cmp.fold == "C_june")].pr_auc.iloc[0])
        self.assertAlmostEqual(E.average_precision_score(va.is_fraud, p), rec, places=6)

if __name__ == "__main__":
    unittest.main()
