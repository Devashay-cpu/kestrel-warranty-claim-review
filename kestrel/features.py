"""Point-in-time feature pipeline. Same code path for training, validation folds, test and the API.

Leakage rules enforced here:
  * label-free history (volume, serial reuse) uses ONLY claims strictly earlier than the claim.
  * label-derived partner history uses ONLY labels of claims submitted <= (t - LAG_DAYS)  ['rolling'],
    or <= a fixed cutoff L                                                              ['frozen'].
  * a claim's own label (and any later label) can never reach its own features.
"""
import numpy as np
import pandas as pd
from . import config as C

GROUPS = {
    "row": ["log_amount", "amount_ratio_list", "list_price_inr", "days_since_purchase", "days_to_expiry",
            "near_expiry", "photo_attached", "partner_inspected", "has_note", "customer_prior_claims",
            "is_small", "auto_approved", "hour", "dow", "family", "partner_type", "note_cat"],
    "partner_volume": ["p_claims_7d", "p_claims_30d", "p_small_share_30d"],
    "partner_label": ["p_lab_n", "p_lab_fraud", "p_lab_n90", "p_lab_fraud90", "p_rate_eb", "p_rate_eb90"],
    # optional groups (off by default; Phase 2 ablates them)
    "partner_age": ["partner_age_days"],
    "serial": ["serial_format", "serial_lower", "serial_prior_uses"],
    "desc": ["desc_cat"],
}
DEFAULT_GROUPS = ["row", "partner_volume", "partner_label"]
CAT_FEATURES = ["family", "partner_type", "note_cat", "serial_format", "desc_cat"]
META = ["claim_id", "submitted_at", "partner_id", "regime_post_may"]
FORBIDDEN = {"claim_id", "source", "is_fraud", "onboarded_date", "submitted_at", "claim_description",
             "inspector_note", "product_serial", "partner_id", "city", "n_submissions", "regime_post_may",
             "label_status", "serial_norm"}


def feature_names(groups=None):
    out = []
    for g in (groups or DEFAULT_GROUPS):
        out += GROUPS[g]
    return out


class _Index:
    """Per-key sorted event times with cumulative weights -> O(log n) window queries."""
    def __init__(self, keys, times, weights=None):
        keys = np.asarray(keys); times = np.asarray(times, dtype="datetime64[ns]")
        w = np.ones(len(keys)) if weights is None else np.asarray(weights, dtype=float)
        self.d = {}
        for k in pd.unique(keys):
            m = keys == k
            t, ww = times[m], w[m]
            o = np.argsort(t, kind="stable")
            self.d[k] = (t[o], np.concatenate([[0.0], np.cumsum(ww[o])]), np.concatenate([[0.0], np.cumsum(np.ones(len(ww)))]))

    def query(self, keys, q_times, window_days=None, side="left"):
        """side='left': events in [q-W, q) (strictly before q).  side='right': events in (q-W, q]."""
        q_times = np.asarray(q_times, dtype="datetime64[ns]")
        n = len(q_times); cnt = np.zeros(n); wsum = np.zeros(n)
        for i, (k, q) in enumerate(zip(keys, q_times)):
            if k not in self.d:
                continue
            t, cw, cc = self.d[k]
            r = np.searchsorted(t, q, side=side)
            lo = 0 if window_days is None else np.searchsorted(t, q - np.timedelta64(window_days, "D"), side=side)
            cnt[i] = cc[r] - cc[lo]; wsum[i] = cw[r] - cw[lo]
        return cnt, wsum


def build_features(df, partners, products, log, events, label_mode=("rolling",), groups=None):
    """df: prepared rows (clean.prepare_rows). log: all claims (label-free). events: labelled claims
    (partner_id, submitted_at, is_fraud) usable as history. label_mode: ('rolling',) or ('frozen', cutoff)."""
    d = df.copy().reset_index(drop=True)
    d = d.merge(partners[["partner_id", "city", "onboarded_date", "partner_type"]], on="partner_id", how="left", validate="m:1")
    d = d.merge(products, on="sku", how="left", validate="m:1")
    assert d.partner_type.notna().all() and d.list_price_inr.notna().all(), "unmatched partner/sku"
    t = d["submitted_at"]

    # ---- row-level ----
    d["log_amount"] = np.log1p(d.claim_amount_inr)
    d["amount_ratio_list"] = d.claim_amount_inr / d.list_price_inr
    d["days_to_expiry"] = d.warranty_months * 30.4375 - d.days_since_purchase
    d["near_expiry"] = (d.days_to_expiry <= 30).astype(int)
    d["photo_attached"] = (d.photo_attached == "Y").astype(int)
    d["partner_inspected"] = (d.partner_inspected == "Y").astype(int)
    d["is_small"] = (d.claim_amount_inr < C.SMALL_CLAIM_INR).astype(int)
    d["auto_approved"] = ((d.is_small == 1) & (t >= C.REGIME_DATE)).astype(int)   # policy s5, derived from date+amount
    d["hour"] = t.dt.hour; d["dow"] = t.dt.dayofweek
    d["partner_age_days"] = (t - d.onboarded_date).dt.days.clip(lower=0)

    # ---- label-free history: strictly earlier claims ----
    lg = log.reset_index(drop=True)
    pidx = _Index(lg.partner_id, lg.submitted_at, lg.is_small)
    c7, _ = pidx.query(d.partner_id, t, 7, "left")
    c30, s30 = pidx.query(d.partner_id, t, 30, "left")
    d["p_claims_7d"], d["p_claims_30d"] = c7, c30
    d["p_small_share_30d"] = np.where(c30 > 0, s30 / np.maximum(c30, 1), 0.0)
    sidx = _Index(lg.serial_norm, lg.submitted_at)
    d["serial_prior_uses"], _ = sidx.query(d.serial_norm, t, None, "left")

    # ---- label-derived partner history (point-in-time) ----
    ev = events.reset_index(drop=True)
    if label_mode[0] == "rolling":
        q = t - pd.Timedelta(days=C.LAG_DAYS)
    else:
        q = pd.Series(pd.Timestamp(label_mode[1]), index=d.index)
    pe = _Index(ev.partner_id, ev.submitted_at, ev.is_fraud)
    ge = _Index(np.zeros(len(ev)), ev.submitted_at, ev.is_fraud)
    n, f = pe.query(d.partner_id, q, None, "right")
    n9, f9 = pe.query(d.partner_id, q, C.WINDOW_DAYS, "right")
    gn, gf = ge.query(np.zeros(len(d)), q, None, "right")
    g = (gf + C.GLOBAL_PRIOR * 50) / (gn + 50)
    d["p_lab_n"], d["p_lab_fraud"], d["p_lab_n90"], d["p_lab_fraud90"] = n, f, n9, f9
    d["p_rate_eb"] = (f + C.EB_STRENGTH * g) / (n + C.EB_STRENGTH)
    d["p_rate_eb90"] = (f9 + C.EB_STRENGTH * g) / (n9 + C.EB_STRENGTH)

    keep = META + sorted({c for g_ in GROUPS.values() for c in g_}) + (["is_fraud"] if "is_fraud" in d else [])
    out = d[keep].copy()
    for c in CAT_FEATURES:
        out[c] = out[c].astype(str)
    return out
