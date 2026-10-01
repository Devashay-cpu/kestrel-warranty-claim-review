"""Validate predictions.csv.  python scripts/validate_submission.py [predictions.csv]   (exit code 1 on any failure)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def validate(pred_path=ROOT / "predictions.csv", sample=ROOT / "data/raw/sample_submission.csv", test=ROOT / "data/raw/test_unlabelled.csv"):
    errs = []
    p = pd.read_csv(pred_path); s = pd.read_csv(sample); t = pd.read_csv(test, usecols=["claim_id"])
    if list(p.columns) != ["claim_id", "score"]: errs.append(f"columns must be exactly claim_id,score; got {list(p.columns)}")
    if list(p.columns) == list(s.columns) == ["claim_id", "score"]:
        if len(p) != len(t): errs.append(f"row count {len(p)} != test rows {len(t)}")
        if p.claim_id.duplicated().any(): errs.append("duplicate claim_id")
        if p.claim_id.isna().any(): errs.append("missing claim_id")
        if set(p.claim_id) != set(t.claim_id): errs.append("claim_id set differs from test_unlabelled.csv")
        if len(p) == len(s) and not (p.claim_id.values == s.claim_id.values).all(): errs.append("order differs from sample_submission.csv")
        if not pd.api.types.is_numeric_dtype(p.score): errs.append("score not numeric")
        elif not np.isfinite(p.score).all(): errs.append("non-finite or missing score")
        elif p.score.nunique() < 50: errs.append("scores nearly constant")
    return errs

if __name__ == "__main__":
    e = validate(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "predictions.csv")
    print("OK: predictions.csv is valid" if not e else "FAILED:\n- " + "\n- ".join(e)); sys.exit(1 if e else 0)
