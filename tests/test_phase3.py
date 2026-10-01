"""Phase 3 tests: final model, predictions.csv, reasons, git-ignore of raw data.  python -m unittest tests.test_phase3 -v
Needs artifacts/phase3/final_model.joblib and predictions.csv from scripts/run_phase3.py."""
import json, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
from kestrel import config as C, features as F, final as FIN
from validate_submission import validate

OUT = C.ART / "phase3"; PRED = ROOT / "predictions.csv"
ok = (OUT / "final_model.joblib").exists() and PRED.exists()

@unittest.skipUnless(ok, "run scripts/run_phase3.py first")
class TestFinalModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = FIN.load_bundle(); cls.pred = pd.read_csv(PRED)
        cls.test = pd.read_csv(C.RAW / "test_unlabelled.csv"); cls.sample = pd.read_csv(C.RAW / "sample_submission.csv")

    def test_model_loads_and_is_the_chosen_one(self):
        b = self.b
        self.assertEqual(b["features"], F.feature_names()); self.assertFalse(set(b["features"]) & F.FORBIDDEN)
        self.assertEqual(type(b["pipeline"].named_steps["clf"]).__name__, "LogisticRegression")
        self.assertAlmostEqual(b["pipeline"].named_steps["clf"].C, 0.5)
        self.assertIn(b["weighting"], ("uniform", "x5"))

    def test_no_test_period_data_in_label_history(self):
        self.assertLess(self.b["events"].submitted_at.max(), pd.Timestamp("2026-07-01"))
        self.assertEqual(self.b["events"].submitted_at.max(), self.b["label_cutoff"])
        self.assertTrue(self.b["events"].is_fraud.isin([0, 1]).all())

    def test_prediction_count_and_ids(self):
        p = self.pred
        self.assertEqual(len(p), len(self.test)); self.assertTrue(p.claim_id.is_unique); self.assertFalse(p.claim_id.isna().any())
        self.assertTrue((p.claim_id.values == self.test.claim_id.values).all())      # order preserved
        self.assertTrue((p.claim_id.values == self.sample.claim_id.values).all())

    def test_scores_numeric_finite_in_range(self):
        s = self.pred.score
        self.assertTrue(pd.api.types.is_numeric_dtype(s)); self.assertTrue(np.isfinite(s).all())
        self.assertTrue(((s >= 0) & (s <= 1)).all()); self.assertGreater(s.nunique(), 1000)

    def test_exact_columns_and_validator(self):
        self.assertEqual(list(self.pred.columns), ["claim_id", "score"]); self.assertEqual(validate(PRED), [])

    def test_validator_catches_bad_files(self):
        with tempfile.TemporaryDirectory() as d:
            for name, df in {"extra": self.pred.assign(x=1), "dup": pd.concat([self.pred.iloc[:-1], self.pred.iloc[[0]]]),
                             "nan": self.pred.assign(score=np.where(np.arange(len(self.pred)) == 5, np.nan, self.pred.score)),
                             "short": self.pred.iloc[:-5], "reordered": self.pred.iloc[::-1]}.items():
                f = Path(d) / f"{name}.csv"; df.to_csv(f, index=False); self.assertTrue(validate(f), name)

    def test_single_record_scoring_matches_file(self):       # same path the API will use
        for i in (0, 777, 2251):
            _, p = FIN.score(self.test.iloc[[i]], self.b)
            self.assertAlmostEqual(float(p[0]), float(self.pred.score.iloc[i]), places=6)

    def test_reasons_are_supported_and_exact(self):
        X, p = FIN.score(self.test.iloc[:200], self.b)
        c = FIN.contributions(self.b, X)
        lin = self.b["pipeline"].named_steps["clf"]; logit = np.log(p / (1 - p))
        const = logit - c.sum(axis=1).values
        self.assertLess(np.ptp(const), 1e-6)                  # contributions + one constant == model logit
        for rs in FIN.reasons(self.b, X.iloc[:50]):
            self.assertLessEqual(len(rs), 3)
            for r in rs: self.assertIn(r["group"], FIN.REASON_GROUPS); self.assertTrue(r["text"]); self.assertGreater(r["effect"], 0)

    def test_artifacts_consistent(self):
        imp = pd.read_csv(OUT / "business_impact.csv")
        self.assertTrue((imp.reviewed_3_months == 120).all()); self.assertTrue((imp.reviewed_per_month == 40).all())
        self.assertTrue(np.allclose(imp.net_benefit_inr, imp.fraud_inr_stopped - imp.goodwill_cost_inr))
        e = json.load(open(OUT / "expected_score_range.json")); self.assertLess(e["low"], e["central"]); self.assertLess(e["central"], e["high"])
        top = pd.read_csv(OUT / "test_top_claims.csv"); self.assertEqual(len(top), 120); self.assertTrue(top.claim_id.isin(self.test.claim_id).all())
        self.assertEqual(top.groupby("month").size().max(), 40)

    def test_weighting_decision_made_before_test_is_scored(self):
        src = (ROOT / "scripts/run_phase3.py").read_text()
        self.assertLess(src.index("USE_X5 = all("), src.index("FIN.score(te_raw"))

class TestGitIgnore(unittest.TestCase):
    def test_raw_data_and_artifacts_ignored(self):
        if shutil.which("git") is None: self.skipTest("git not available")
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q", d], check=True); shutil.copy(ROOT / ".gitignore", Path(d) / ".gitignore")
            for f in ("data/raw/train.csv", "data/raw/test_unlabelled.csv", "data/raw/partners.csv", "artifacts/phase3/final_model.joblib", "artifacts/features_train.csv"):
                r = subprocess.run(["git", "-C", d, "check-ignore", "-q", f]); self.assertEqual(r.returncode, 0, f"{f} is NOT ignored")

if __name__ == "__main__":
    unittest.main()
