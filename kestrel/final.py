"""Load the saved final model and score raw claim records (same code path as Phase 1/2 features). Used by Phase 3 and the later API."""
import joblib
import numpy as np
import pandas as pd
from . import config as C, features as F
from .clean import prepare_rows

BUNDLE_PATH = C.ART / "phase3" / "final_model.joblib"

# plain-language reason groups: only things the model really uses
REASON_GROUPS = {
    "customer_history": ["customer_prior_claims"],
    "partner_fraud_history": ["p_lab_n", "p_lab_fraud", "p_lab_n90", "p_lab_fraud90", "p_rate_eb", "p_rate_eb90"],
    "partner_activity": ["p_claims_7d", "p_claims_30d", "p_small_share_30d"],
    "small_uninspected": ["is_small", "auto_approved", "partner_inspected", "has_note", "note_cat"],
    "amount": ["log_amount", "amount_ratio_list", "list_price_inr"],
    "timing": ["days_since_purchase", "days_to_expiry", "near_expiry", "hour", "dow"],
    "photo": ["photo_attached"],
    "product_and_outlet_type": ["family", "partner_type"],
}
FEATURE_TO_GROUP = {f: g for g, fs in REASON_GROUPS.items() for f in fs}


def load_bundle(path=BUNDLE_PATH):
    return joblib.load(path)


def featurize(raw: pd.DataFrame, b) -> pd.DataFrame:
    """raw claim rows (test_unlabelled schema) -> model features. Partner label history frozen at the bundle's label cutoff."""
    d = prepare_rows(raw)
    d["regime_post_may"] = (d.submitted_at >= C.REGIME_DATE).astype(int)   # metadata only, never a model input
    return F.build_features(d, b["partners"], b["products"], b["log"], b["events"], ("frozen", b["label_cutoff"]))


def score(raw: pd.DataFrame, b):
    X = featurize(raw, b)
    return X, b["pipeline"].predict_proba(X[b["features"]])[:, 1]


def _dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m)


def contributions(b, X: pd.DataFrame) -> pd.DataFrame:
    """Per-claim log-odds contribution of each reason group relative to an average training claim.
    sum(groups) + average-claim logit == model logit (exact for logistic regression)."""
    pre, clf = b["pipeline"].named_steps["pre"], b["pipeline"].named_steps["clf"]
    Z = _dense(pre.transform(X[b["features"]])) - b["baseline"]
    contrib = Z * clf.coef_[0]
    out = pd.DataFrame(0.0, index=X.index, columns=list(REASON_GROUPS))
    for j, src in enumerate(b["out_to_src"]):
        out[FEATURE_TO_GROUP[src]] += contrib[:, j]
    return out


def _text(g, r):
    amt = float(np.expm1(r["log_amount"]))
    if g == "customer_history":
        return f"Customer has {int(r['customer_prior_claims'])} earlier warranty claim(s)"
    if g == "partner_fraud_history":
        return (f"Outlet {r['partner_id']} has {int(r['p_lab_fraud'])} confirmed fraud case(s) among {int(r['p_lab_n'])} decided claims "
                f"({int(r['p_lab_fraud90'])} in the last 90 days)")
    if g == "partner_activity":
        return f"Outlet filed {int(r['p_claims_30d'])} claims in the last 30 days, {r['p_small_share_30d']:.0%} of them under Rs {C.SMALL_CLAIM_INR:,}"
    if g == "small_uninspected":
        if r["auto_approved"] == 1:
            return f"Small claim (Rs {amt:,.0f}, under Rs {C.SMALL_CLAIM_INR:,}) approved without inspection"
        return "Inspection status / inspector note pattern has been linked to more fraud in past data"
    if g == "amount":
        return f"Claim amount Rs {amt:,.0f} ({r['amount_ratio_list']:.0%} of list price) is in a range linked to more fraud in past data"
    if g == "timing":
        return f"Timing: filed {int(r['days_since_purchase'])} days after purchase ({r['days_to_expiry']:.0f} days of warranty left), a pattern linked to more fraud in past data"
    if g == "photo":
        return "No photo attached" if r["photo_attached"] == 0 else "Photo-attachment pattern linked to more fraud in past data"
    return f"Product family / outlet type ({r['family']}, {r['partner_type'].replace('_', ' ')}) linked to more fraud in past data"


def reasons(b, X: pd.DataFrame, top=3, min_effect=0.05):
    """List (per row) of up to `top` reasons that RAISE the risk, strongest first: dict(group, effect, text)."""
    C_ = contributions(b, X)
    res = []
    for i in range(len(X)):
        row = X.iloc[i]; c = C_.iloc[i].sort_values(ascending=False)
        res.append([dict(group=g, effect=float(v), text=_text(g, row)) for g, v in c.items() if v >= min_effect][:top])
    return res
