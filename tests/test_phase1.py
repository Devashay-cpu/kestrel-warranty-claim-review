"""Data-quality, leakage and reuse tests.  Run:  python -m unittest discover -s tests -v   (needs data/raw/*.csv)"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from kestrel import config as C, features as F, splits as S, pipeline as P
from kestrel.clean import load_all

ds = XT = XE = None
def setUpModule():
    global ds, XT, XE
    ds = load_all(); XT = P.train_features(ds); XE = P.test_features(ds)

ALL = F.feature_names(list(F.GROUPS))

class TestCleaning(unittest.TestCase):
    def test_unique_ids(self):
        for k in ("train", "model_set", "test"):
            self.assertTrue(ds[k].claim_id.is_unique, k)
        self.assertEqual(ds["report"]["id_overlap_train_test"], 0)

    def test_blank_labels_accounted(self):
        raw = pd.read_csv(C.RAW / "train.csv")
        self.assertEqual(ds["report"]["undecided_claims"], raw[raw.is_fraud.isna()].claim_id.nunique())
        self.assertEqual(len(ds["model_set"]) + len(ds["undecided"]) + len(ds["excluded"]), len(ds["train"]))
        self.assertTrue(ds["model_set"].is_fraud.isin([0, 1]).all())
        self.assertEqual(ds["undecided"].source.unique().tolist(), ["crm"])

    def test_dedupe_consistent(self):
        self.assertEqual(ds["report"]["dedupe_label_conflicts_between_resubmits"], 0)
        self.assertEqual(len(ds["train"]), 11348)

    def test_injected_text_excluded_and_not_used(self):
        ids = set(ds["report"]["injected_text_claim_ids"])
        self.assertEqual(len(ids), 4)
        self.assertFalse(ids & set(ds["model_set"].claim_id))
        self.assertEqual(ds["report"]["test_suspect_description_rows"], 0)
        self.assertTrue(set(XT.desc_cat) <= set(C.ALLOWED_DESCRIPTIONS) | {"other"})

    def test_no_bad_values(self):
        m = ds["model_set"]
        self.assertTrue((m.claim_amount_inr > 0).all() and (m.days_since_purchase >= 0).all())
        self.assertTrue(m.serial_norm.str.len().eq(10).all() or m.serial_norm.str.len().between(9, 11).all())

class TestFeatureSchema(unittest.TestCase):
    def test_no_forbidden_columns_used(self):
        self.assertFalse(set(ALL) & F.FORBIDDEN)

    def test_same_columns_train_test(self):
        feats = F.feature_names(list(F.GROUPS))
        self.assertTrue(set(feats) <= set(XT.columns) and set(feats) <= set(XE.columns))
        self.assertEqual(len(XE), 2252)

    def test_no_nulls(self):
        for X in (XT, XE):
            self.assertEqual(int(X[ALL].isna().sum().sum()), 0)

    def test_leak_canary(self):
        # a point-in-time partner feature must not be near-perfect; a leak would push AUC towards 1
        for c in ("p_rate_eb", "p_rate_eb90", "p_lab_fraud"):
            self.assertLess(roc_auc_score(XT.is_fraud, XT[c]), 0.95, c)

class TestRegime(unittest.TestCase):
    def test_auto_approved_matches_policy(self):
        for X in (XT, XE):
            exp = ((X.is_small == 1) & (X.submitted_at >= C.REGIME_DATE)).astype(int)
            self.assertTrue((X.auto_approved == exp).all())
        # in the test period every small claim is uninspected
        te = ds["test"]
        self.assertTrue((te[te.claim_amount_inr < C.SMALL_CLAIM_INR].partner_inspected == "N").all())
        self.assertTrue((XE.regime_post_may == 1).all())

class TestPointInTime(unittest.TestCase):
    T0 = pd.Timestamp("2026-03-01")

    def _rebuild(self, ms_events, log):
        ms = ds["model_set"]
        return F.build_features(ms, ds["partners"], ds["products"], log, ms_events, ("rolling",))

    def test_future_labels_cannot_change_past_features(self):
        ev = P.events_of(ds["model_set"]).copy()
        ev.loc[ev.submitted_at > self.T0 - pd.Timedelta(days=C.LAG_DAYS), "is_fraud"] ^= 1   # flip every recent/future label
        X2 = self._rebuild(ev, ds["log"])
        past = XT.submitted_at <= self.T0
        cols = F.GROUPS["partner_label"]
        pd.testing.assert_frame_equal(XT.loc[past, cols].reset_index(drop=True), X2.loc[past, cols].reset_index(drop=True))
        self.assertFalse(XT.loc[~past, cols].reset_index(drop=True).equals(X2.loc[~past, cols].reset_index(drop=True)))  # test is sensitive

    def test_own_label_never_in_own_features(self):
        ev = P.events_of(ds["model_set"]).copy(); ev["is_fraud"] ^= 1                      # flip ALL labels
        X2 = self._rebuild(ev, ds["log"])
        first = XT.groupby("partner_id").submitted_at.transform("min") == XT.submitted_at   # each partner's first claim
        self.assertTrue((XT.loc[first, "p_lab_fraud"] == 0).all() and (X2.loc[first, "p_lab_fraud"] == 0).all())

    def test_volume_features_use_only_earlier_claims(self):
        lg = ds["log"]; lg2 = lg[lg.submitted_at < self.T0]
        X2 = F.build_features(ds["model_set"][ds["model_set"].submitted_at < self.T0], ds["partners"], ds["products"], lg2,
                              P.events_of(ds["model_set"]), ("rolling",))
        cols = F.GROUPS["partner_volume"] + ["serial_prior_uses"]
        a = XT[XT.submitted_at < self.T0][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, X2[cols].reset_index(drop=True))

    def test_frozen_val_features_ignore_labels_after_cutoff(self):
        fold = S.FOLDS[1]; ms = ds["model_set"]; cut = S.fold_cutoff(fold)
        _, v1 = P.fold_data(ds, fold, XT)                                   # events <= cutoff only
        ev = P.events_of(ms); ev.loc[ev.submitted_at > cut, "is_fraud"] ^= 1   # flip every label after the cutoff
        _, va = S.fold_masks(ms, fold)
        v2 = F.build_features(ms[va], ds["partners"], ds["products"], ds["log"], ev, ("frozen", cut))
        cols = F.GROUPS["partner_label"]
        pd.testing.assert_frame_equal(v1[cols].reset_index(drop=True), v2[cols].reset_index(drop=True))

    def test_single_record_equals_batch(self):             # API reuse
        ms = ds["model_set"]; te = ds["test"]
        ev = P.events_of(ms); cut = ms.submitted_at.max()
        for i in (0, 700, 2251):
            one = F.build_features(te.iloc[[i]], ds["partners"], ds["products"], ds["log"], ev, ("frozen", cut))
            pd.testing.assert_frame_equal(one[ALL].reset_index(drop=True), XE.loc[[i], ALL].reset_index(drop=True), check_dtype=False)

class TestFolds(unittest.TestCase):
    def test_folds_are_time_ordered_with_embargo(self):
        ms = ds["model_set"]
        for fold in S.FOLDS:
            tr, va = S.fold_masks(ms, fold)
            self.assertFalse((tr & va).any())
            self.assertLess(ms[tr].submitted_at.max(), ms[va].submitted_at.min())
            gap = ms[va].submitted_at.min() - ms[tr].submitted_at.max()
            self.assertGreaterEqual(gap, pd.Timedelta(days=C.LAG_DAYS - 1))
            self.assertGreater(ms[va].is_fraud.sum(), 10)          # enough positives to evaluate
            self.assertEqual(len(set(ms[tr].claim_id) & set(ms[va].claim_id)), 0)

if __name__ == "__main__":
    unittest.main()
