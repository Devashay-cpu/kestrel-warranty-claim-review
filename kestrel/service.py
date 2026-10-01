"""Phase 4: framework-free service logic (validation -> Phase 3 scoring -> risk level + reasons).

app.py is only a thin FastAPI wrapper around this module, so everything important can be tested
without a web server. Nothing here retrains or changes the Phase 3 model: it calls kestrel.final
(`score`, `reasons`) on the saved bundle exactly as scripts/run_phase3.py did for predictions.csv.
"""
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import config as C, features as F, final as FIN
from .clean import normalise_serial

# ---- risk levels (a triage label for the investigation desk, NOT a decision) ----
# Cut-offs come from the Phase 3 score distribution on the 2,252 test claims (predictions.csv):
#   score >= 0.13  -> about the top 5% of claims   (desk capacity is 40 of ~750 claims a month = 5.3%)  -> "high"
#   score >= 0.04  -> about the top 10% of claims                                                       -> "medium"
#   below 0.04     -> the other ~90%                                                                    -> "low"
# tests/test_phase4_service.py re-checks these shares against predictions.csv.
HIGH_THRESHOLD = 0.13
MEDIUM_THRESHOLD = 0.04
RISK_NOTE = "High score means review first, not automatic rejection."
NO_REASON_TEXT = "No strong single factor identified; review the overall score."
MAX_REASONS = 3

IST = timezone(timedelta(hours=5, minutes=30))     # raw data is IST wall-clock time; India has no DST
REQUIRED = ["submitted_at", "partner_id", "sku", "claim_amount_inr", "days_since_purchase",
            "photo_attached", "partner_inspected", "customer_prior_claims"]
OPTIONAL = ["claim_id", "product_serial", "claim_description", "inspector_note"]
PARTNER_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class ClaimError(ValueError):
    """The request is not a usable claim. `.errors` is a list of {field, problem} dicts (safe to show to the user)."""
    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(f"{e['field']}: {e['problem']}" for e in errors))


def risk_level(score: float) -> str:
    return "high" if score >= HIGH_THRESHOLD else "medium" if score >= MEDIUM_THRESHOLD else "low"


# ----------------------------------------------------------------------------- validation
def _number(v, field, lo, hi, errs, integer=False, lo_open=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        errs.append(dict(field=field, problem="must be a number")); return None
    if not math.isfinite(v):
        errs.append(dict(field=field, problem="must be a finite number")); return None
    if integer and v != int(v):
        errs.append(dict(field=field, problem="must be a whole number")); return None
    if (v <= lo if lo_open else v < lo) or v > hi:
        errs.append(dict(field=field, problem=f"must be {'greater than' if lo_open else 'at least'} {lo:g} and at most {hi:g}")); return None
    return int(v) if integer else float(v)


def _yes_no(v, field, errs):
    if isinstance(v, bool):
        return "Y" if v else "N"
    if isinstance(v, str) and v.strip().lower() in ("y", "yes", "true"):
        return "Y"
    if isinstance(v, str) and v.strip().lower() in ("n", "no", "false"):
        return "N"
    errs.append(dict(field=field, problem="must be Y or N")); return None


def _text(v, field, maxlen, errs):
    if v is None:
        return ""
    if not isinstance(v, str):
        errs.append(dict(field=field, problem="must be text")); return None
    if len(v) > maxlen:
        errs.append(dict(field=field, problem=f"must be at most {maxlen} characters")); return None
    return v.strip()


def _timestamp(v, errs):
    if not isinstance(v, str) or not v.strip():
        errs.append(dict(field="submitted_at", problem="must be a date-time text such as 2026-10-02T14:30:00")); return None
    try:
        t = datetime.fromisoformat(v.strip())
    except ValueError:
        errs.append(dict(field="submitted_at", problem="is not a valid ISO date-time; use e.g. 2026-10-02T14:30:00")); return None
    if t.tzinfo is not None:                     # convert to IST wall-clock, as the training data is stored
        t = t.astimezone(IST).replace(tzinfo=None)
    if not 2000 <= t.year <= 2100:
        errs.append(dict(field="submitted_at", problem="year must be between 2000 and 2100")); return None
    return t


def validate_claim(payload, valid_skus):
    """-> (clean claim dict in the raw test_unlabelled schema, list of non-fatal warnings). Raises ClaimError."""
    if not isinstance(payload, dict):
        raise ClaimError([dict(field="(body)", problem="must be one JSON object describing a single claim")])
    errs, w = [], []
    missing = [f for f in REQUIRED if f not in payload or payload[f] is None or payload[f] == ""]
    for f in missing:
        errs.append(dict(field=f, problem="is required"))
    ok = lambda f: f not in missing
    c = {}
    if ok("submitted_at"): c["submitted_at"] = _timestamp(payload["submitted_at"], errs)
    if ok("partner_id"):
        pid = payload["partner_id"]
        if isinstance(pid, str) and PARTNER_RE.match(pid.strip()):
            c["partner_id"] = pid.strip()
        else:
            errs.append(dict(field="partner_id", problem="must be text of 1-32 letters, digits, '-' or '_' (e.g. SP3118)")); c["partner_id"] = None
    if ok("sku"):
        s = payload["sku"].strip() if isinstance(payload["sku"], str) else None
        if s is None:
            errs.append(dict(field="sku", problem="must be text")); c["sku"] = None
        elif s not in valid_skus:
            errs.append(dict(field="sku", problem=f"unknown product code; known codes: {', '.join(sorted(valid_skus))}")); c["sku"] = None
        else:
            c["sku"] = s
    if ok("claim_amount_inr"): c["claim_amount_inr"] = _number(payload["claim_amount_inr"], "claim_amount_inr", 0, 10_000_000, errs, lo_open=True)
    if ok("days_since_purchase"): c["days_since_purchase"] = _number(payload["days_since_purchase"], "days_since_purchase", 0, 36_500, errs, integer=True)
    if ok("customer_prior_claims"): c["customer_prior_claims"] = _number(payload["customer_prior_claims"], "customer_prior_claims", 0, 1_000, errs, integer=True)
    if ok("photo_attached"): c["photo_attached"] = _yes_no(payload["photo_attached"], "photo_attached", errs)
    if ok("partner_inspected"): c["partner_inspected"] = _yes_no(payload["partner_inspected"], "partner_inspected", errs)
    c["claim_id"] = _text(payload.get("claim_id"), "claim_id", 64, errs) or "API-CLAIM"
    c["product_serial"] = _text(payload.get("product_serial"), "product_serial", 64, errs)
    c["claim_description"] = _text(payload.get("claim_description"), "claim_description", 500, errs)   # never interpreted beyond an allow-list
    c["inspector_note"] = _text(payload.get("inspector_note"), "inspector_note", 500, errs)           # optional
    if errs:
        raise ClaimError(errs)
    ignored = sorted(set(payload) - set(REQUIRED) - set(OPTIONAL))
    if ignored:
        w.append("Ignored fields the model does not use: " + ", ".join(str(k)[:40] for k in ignored[:10]))
    return c, w


# ----------------------------------------------------------------------------- scoring
class Predictor:
    """Holds the saved Phase 3 bundle (loaded once) and scores one claim at a time."""

    def __init__(self, bundle):
        need = {"pipeline", "features", "baseline", "out_to_src", "label_cutoff", "events", "log", "partners", "products"}
        if not isinstance(bundle, dict) or need - set(bundle):
            raise RuntimeError(f"model bundle is missing parts: {sorted(need - set(bundle)) if isinstance(bundle, dict) else 'not a dict'}")
        if list(bundle["features"]) != F.feature_names():
            raise RuntimeError("model bundle features do not match kestrel.features.feature_names(); rebuild with scripts/run_phase3.py")
        self.b = bundle
        self.skus = set(bundle["products"].sku)
        self.known_partners = set(bundle["partners"].partner_id)
        self.history_end = pd.Timestamp(bundle["log"].submitted_at.max())
        self.label_cutoff = pd.Timestamp(bundle["label_cutoff"])

    @classmethod
    def from_path(cls, path=None):
        return cls(FIN.load_bundle(path or FIN.BUNDLE_PATH))

    def meta(self):
        return dict(skus=sorted(self.skus), allowed_inspector_notes=C.ALLOWED_NOTES, allowed_descriptions=C.ALLOWED_DESCRIPTIONS,
                    risk_thresholds=dict(high=HIGH_THRESHOLD, medium=MEDIUM_THRESHOLD), risk_note=RISK_NOTE,
                    model=dict(type="LogisticRegression", weighting=self.b.get("weighting"), label_history_until=str(self.label_cutoff),
                               claim_activity_history_until=str(self.history_end)))

    def _bundle_for(self, claim, partner_known):
        """Unseen outlet: add it to a COPY of the partner table as an outlet of unknown type (no history exists for it).
        The one-hot encoder ignores the unknown type, and every history count is 0, exactly like a brand-new outlet."""
        if partner_known:
            return self.b
        row = pd.DataFrame([dict(partner_id=claim["partner_id"], city="unknown", onboarded_date=pd.Timestamp(claim["submitted_at"]),
                                 partner_type="unknown")])
        return {**self.b, "partners": pd.concat([self.b["partners"], row], ignore_index=True)}

    def _relevant_log(self, claim):
        """Speed only: features look up history for THIS claim's outlet and serial, so rows of other outlets/serials can never
        change the result. Dropping them turns a ~1.6 s request into a fast one; tests prove the score is identical."""
        lg = self.b["log"]
        return lg[(lg.partner_id == claim["partner_id"]) | (lg.serial_norm == normalise_serial(claim["product_serial"]))]

    def predict(self, payload):
        claim, warnings = validate_claim(payload, self.skus)
        known = claim["partner_id"] in self.known_partners
        t = pd.Timestamp(claim["submitted_at"])
        if not known:
            warnings.append(f"Outlet {claim['partner_id']} is not in the outlet list the model was built on. It is scored as an outlet with "
                            "no history, so the outlet-based part of the score is missing.")
        if t < self.label_cutoff:
            warnings.append(f"Claim date is before {self.label_cutoff.date()}: outlet fraud history is taken as of that date, so it includes cases that "
                            "were not yet known on the claim date.")
        if t > self.history_end + pd.Timedelta(days=1):
            warnings.append(f"Outlet activity counts use claims up to {self.history_end.date()} only; activity after that date is not included.")
        b = self._bundle_for(claim, known)
        b = {**b, "log": self._relevant_log(claim)}
        X, p = FIN.score(pd.DataFrame([claim]), b)       # same function that wrote predictions.csv
        score = float(p[0])
        if not math.isfinite(score):
            raise RuntimeError("model returned a non-finite score")
        reasons = [r["text"] for r in FIN.reasons(b, X, top=MAX_REASONS)[0]][:MAX_REASONS]
        return dict(claim_id=claim["claim_id"], fraud_score=round(score, 8), risk_level=risk_level(score),
                    reasons=reasons or [NO_REASON_TEXT], partner_known=known, warnings=warnings, note=RISK_NOTE)
