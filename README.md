# Kestrel Home – warranty fraud review

A small service that gives **one warranty claim** a fraud *review-priority* score, a risk level, and up to three plain-language reasons.
It is built on the model from Phases 1-3 of this project (reports in `docs/`).

> **This is a review-priority tool. A high score is not a fraud verdict. High score means review first, not automatic rejection.**

## What problem it solves
Kestrel's investigation desk can check about **40 claims a month** out of roughly 750, and only about 1-3% of claims are fraud. The model
ranks claims so the desk looks at the most suspicious ones first, and says why. It never approves, rejects or delays a customer.
Plain accuracy is useless here: a model that flags nothing is "97-99% accurate", so we judge the model on how well it *ranks* fraud (PR-AUC) and on
rupees stopped per claim checked (see `docs/MEMO_TO_RITU.md`).

## Model, validation, expected score
- **Model:** logistic regression (C=0.5) on 26 features: claim amount and timing, small-claim/auto-approval rule (claims under Rs 2,000 are approved
  without inspection since 1 May 2026), customer's earlier claims, and each outlet's recent claim volume and past confirmed-fraud history.
  Claims filed after May count 5x in training because the approval rule changed. Trained on 11,142 decided claims.
- **Validation:** time-ordered folds only (train on the past, test on a later period), outlet history frozen the way it will be at scoring time,
  and outlet-grouped bootstrap intervals. No random shuffling.
- **Expected hidden-test score (an estimate, not a result, because the hidden outcomes are not available):** PR-AUC **about 0.28, plausible range
  0.17-0.52**. If new fraud starts at outlets we have no history for, it could fall to about 0.03, which is roughly chance. Each scenario's interval is wide (about 0.04 to 0.80).
- **Known limitations:** it cannot see fraud at outlets with no history until a few cases are investigated; the 7 outlets behind most of the queue
  drive the result; the Rs 1,950-1,999 threshold-gaming pattern is not modelled; the economics are thin if the Rs 260 contact cost applies.
  Full detail: `docs/PHASE3_EVIDENCE_REPORT.md`.

## Install (Python 3.12)
```bash
python -m venv .venv
# macOS/Linux:  source .venv/bin/activate        Windows PowerShell:  .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
No API key, paid service or internet connection is needed to run it (internet is only needed once, for `pip install`).

### The saved model
The service loads `artifacts/phase3/final_model.joblib` once at startup; it never retrains. That file embeds Kestrel's claim history, so it is
**client-confidential and git-ignored** (like everything in `data/` and `artifacts/`). To create it on a new machine, put the client files in
`data/raw/` (`train.csv`, `test_unlabelled.csv`, `partners.csv`, `products.csv`, `sample_submission.csv`) and run the earlier phases:
```bash
python scripts/run_phase1.py && python scripts/run_phase2.py && python scripts/run_phase3.py
```
Or copy a ready-made `final_model.joblib` to `artifacts/phase3/` (or set `KESTREL_BUNDLE=/path/to/final_model.joblib`).
If the file is missing the service still starts, `GET /health` says `"model_loaded": false`, and `/predict` returns 503.

## Run
```bash
uvicorn app:app --reload
```
- **UI:** open <http://127.0.0.1:8000/> - fill in the claim (or press *Fill example*) and press **Check Claim**.
- **API docs:** <http://127.0.0.1:8000/docs>

## API
`GET /health` -> `{"status": "ok", "model_loaded": true}`

`POST /predict` - one claim as JSON.

| field | required | notes |
|---|---|---|
| `submitted_at` | yes | ISO date-time, e.g. `2026-10-02T14:30:00` (times with a timezone are converted to IST) |
| `partner_id` | yes | outlet id; an outlet the model has never seen is allowed (see below) |
| `sku` | yes | product code; unknown codes give a 422 listing the valid ones |
| `claim_amount_inr` | yes | number > 0 |
| `days_since_purchase` | yes | whole number >= 0 |
| `customer_prior_claims` | yes | whole number >= 0 |
| `photo_attached`, `partner_inspected` | yes | `Y` or `N` |
| `inspector_note` | no | may be omitted, null or empty |
| `claim_id`, `product_serial`, `claim_description` | no | not used for scoring beyond a fixed allow-list; free text can never change the score |

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "claim_id": "DEMO-1", "submitted_at": "2026-10-02T14:30:00", "partner_id": "SP9001", "sku": "KH-AF-01",
  "claim_amount_inr": 1500, "days_since_purchase": 200, "photo_attached": "N", "partner_inspected": "N",
  "customer_prior_claims": 2 }'
```
Response (real output from the saved model; `SP9001` is a made-up outlet, so it is scored as "no history"):
```json
{
  "claim_id": "DEMO-1",
  "fraud_score": 0.17026835,
  "risk_level": "high",
  "reasons": [
    "Small claim (Rs 1,500, under Rs 2,000) approved without inspection",
    "Customer has 2 earlier warranty claim(s)",
    "Outlet SP9001 has 0 confirmed fraud case(s) among 0 decided claims (0 in the last 90 days)"
  ],
  "partner_known": false,
  "warnings": ["Outlet SP9001 is not in the outlet list the model was built on. It is scored as an outlet with no history, so the outlet-based part of the score is missing.", "..."],
  "note": "High score means review first, not automatic rejection."
}
```
**Errors** never return a traceback: invalid JSON -> `400`; a missing or malformed field -> `422` with `{"detail": ..., "errors": [{"field": "...", "problem": "..."}]}`;
body over 64 KB -> `413`; model not loaded -> `503`.

**Risk level** is a triage label, not a decision. `high` = score >= 0.13 (about the top 5% of the Phase 3 test scores, matching the desk's 40-in-750 capacity),
`medium` = >= 0.04 (about the top 10%), otherwise `low`. The thresholds are `HIGH_THRESHOLD` / `MEDIUM_THRESHOLD` in `kestrel/service.py`.

**Reasons** come from the Phase 3 logic (`kestrel/final.py`): the factors that push this claim's score above an average claim, strongest first, at most 3.
If none is strong, the reason is "No strong single factor identified; review the overall score."

**Unseen outlets** are scored as outlets with no history (their type is unknown, all history counts are zero) and the response says so in `warnings`.
**Stateless:** a scored claim is not added to the history, and outlet activity counts use claims up to the model's last data date (the response warns about dates outside it).

## Tests
```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -t .      # or: python -m pytest -q
```
`tests/test_phase4_service.py` and `tests/test_phase4_api.py` use a **synthetic** model built on made-up data (`tests/synthetic_bundle.py`), so they need no client data.
Extra checks against the real saved model run only when it is present. The Phase 1-3 tests need the raw client files in `data/raw/`.

## Data privacy
Client data must never be committed: `data/`, `artifacts/`, virtual environments and caches are in `.gitignore`. The README, source, tests and UI contain only
synthetic values (`SP9001`, `DEMO-1`) and public product codes. `predictions.csv` (claim id + score) is the only row-level output kept in the repository.

## Layout
`app.py` (FastAPI) - `static/index.html` (UI) - `kestrel/service.py` (validation, scoring, risk level) - `kestrel/final.py` (Phase 3 scoring and reasons) -
`scripts/` (Phases 1-3) - `docs/` (reports) - `tests/`.
