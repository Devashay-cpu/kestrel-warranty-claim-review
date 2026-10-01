# Phase 3 - final fit, predictions, impact

From a clean machine (Python 3.10+; no paid API, no network needed at run time):
```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put train.csv, test_unlabelled.csv, partners.csv, products.csv, sample_submission.csv into data/raw/   (git-ignored client data)
python scripts/run_phase1.py            # cleaning, features, folds        (~15 s)
python scripts/run_phase2.py            # model comparison                 (~3 min)
python scripts/run_phase3.py            # weighting decision, final fit, predictions.csv, impact (~1 min)
python scripts/validate_submission.py   # checks predictions.csv against sample_submission.csv and the test file
python -m unittest discover -s tests -v # 41 tests (~35 s)
```
Phase 3 only (Phase 1-2 artifacts already built): `python scripts/run_phase3.py && python scripts/validate_submission.py`.

Outputs: `predictions.csv` (claim_id,score), `artifacts/phase3/` (final_model.joblib, final_model_summary.json, expected_score_range.json, business_impact.csv, test_prediction_summary.csv, test_top_claims.csv, weighting_*.csv, final_model_coefficients.csv), `docs/PHASE3_EVIDENCE_REPORT.md`, `docs/MEMO_TO_RITU.md`.
Load the model: `from kestrel import final; b = final.load_bundle(); X, p = final.score(raw_df, b); final.reasons(b, X)`.
`data/` and `artifacts/` are git-ignored; do not commit them (ops-policy s10). `predictions.csv` holds only claim ids and scores.
