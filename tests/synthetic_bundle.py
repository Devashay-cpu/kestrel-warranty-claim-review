"""Builds a small SYNTHETIC model bundle (made-up outlets, products and claims, no client data) with the same structure as
artifacts/phase3/final_model.joblib, using the project's own feature and model code. Lets the Phase 4 tests run on a clean checkout."""
import numpy as np
import pandas as pd
from kestrel import config as C, features as F, models as M, final as FIN
from kestrel.clean import prepare_rows

SKUS = ["TEST-A-01", "TEST-A-02", "TEST-B-01"]
KNOWN_PARTNER = "SP0001"
BAD_PARTNER = "SP0002"          # synthetic outlet with several confirmed frauds
CUTOFF = pd.Timestamp("2026-06-30 23:40:00")


def make_bundle(seed=0):
    rng = np.random.default_rng(seed)
    types = ["authorised_service_centre", "franchise", "freelance_technician"]
    partners = pd.DataFrame({"partner_id": [f"SP{i:04d}" for i in range(1, 31)], "city": "Testville",
                             "onboarded_date": pd.Timestamp("2024-01-01"), "partner_type": [types[i % 3] for i in range(30)]})
    products = pd.DataFrame({"sku": SKUS, "family": ["Alpha", "Alpha", "Beta"], "list_price_inr": [4000, 6000, 9000], "warranty_months": [12, 12, 24]})
    n = 1500
    t0 = pd.Timestamp("2026-01-01"); span = (pd.Timestamp("2026-09-30") - t0).total_seconds()
    sub = (t0 + pd.to_timedelta(np.sort(rng.random(n)) * span, unit="s")).round("min")
    pid = rng.choice(partners.partner_id.values, n)
    amt = np.round(rng.lognormal(7.2, 0.9, n)).clip(150, 40000)
    raw = pd.DataFrame({
        "claim_id": [f"SYN{i:06d}" for i in range(n)], "submitted_at": sub, "partner_id": pid,
        "product_serial": [f"KH{rng.integers(10**8, 10**9)}" for _ in range(n)], "sku": rng.choice(SKUS, n),
        "claim_description": "motor not running", "claim_amount_inr": amt, "days_since_purchase": rng.integers(1, 700, n),
        "photo_attached": rng.choice(["Y", "N"], n, p=[.7, .3]), "partner_inspected": rng.choice(["Y", "N"], n, p=[.6, .4]),
        "inspector_note": rng.choice(["", "Unit inspected, fault confirmed"], n), "customer_prior_claims": rng.poisson(0.4, n)})
    p_fraud = 0.01 + 0.25 * (raw.partner_id == BAD_PARTNER) + 0.03 * (raw.customer_prior_claims >= 2) + 0.02 * (raw.claim_amount_inr < C.SMALL_CLAIM_INR)
    raw["is_fraud"] = (rng.random(n) < p_fraud).astype(int)
    lab = raw[raw.submitted_at <= CUTOFF].copy()
    log = prepare_rows(raw)[["claim_id", "partner_id", "submitted_at", "claim_amount_inr", "serial_norm"]].copy()
    log["is_small"] = (log.claim_amount_inr < C.SMALL_CLAIM_INR).astype(int)
    log = log.sort_values(["submitted_at", "claim_id"], kind="stable").reset_index(drop=True)
    events = lab[["partner_id", "submitted_at", "is_fraud"]].sort_values("submitted_at").reset_index(drop=True)
    prep = prepare_rows(lab); prep["regime_post_may"] = (prep.submitted_at >= C.REGIME_DATE).astype(int)
    X = F.build_features(prep, partners, products, log, events, ("rolling",))
    w = np.where(X.regime_post_may.values == 1, 5.0, 1.0)
    feats = F.feature_names()
    _, pipe = M.fit_predict("lr", {"name": "C=0.5", "C": 0.5}, feats, X, X.iloc[:5], weights=w, seed=0)
    pre = pipe.named_steps["pre"]; Z = FIN._dense(pre.transform(X[feats])); cats = [f for f in feats if f in F.CAT_FEATURES]
    def src(name):
        name = name.split("__", 1)[1]
        return name if name in feats else next(c for c in cats if name.startswith(c + "_"))
    return dict(pipeline=pipe, features=feats, params={"name": "C=0.5", "C": 0.5}, weighting="x5", baseline=Z.mean(0),
                out_to_src=[src(x) for x in pre.get_feature_names_out()], label_cutoff=CUTOFF, events=events, log=log, partners=partners,
                products=products, constants={}, train_rows=len(X), train_fraud=int(X.is_fraud.sum()), train_range=[str(lab.submitted_at.min()), str(CUTOFF)])


def valid_claim(**over):
    c = dict(claim_id="SYN-API-1", submitted_at="2026-10-02T14:30:00", partner_id=KNOWN_PARTNER, sku=SKUS[0], claim_amount_inr=1500,
             days_since_purchase=200, photo_attached="N", partner_inspected="N", customer_prior_claims=2, inspector_note=None)
    c.update(over); return c
