# PROJECT_STATUS - Kestrel Home warranty fraud (Task 2, Variant C)

**Current phase:** Phase 4 BUILT. Service logic tested (31 tests pass, real + synthetic model). FastAPI layer, UI and clean-machine run NOT yet executed: the build sandbox had no network, so fastapi/uvicorn/httpx could not be installed. Run the 'Phase 4 verification still to do' list below, then mark Phase 4 COMPLETE.
**Last updated:** 2026-10-02

## Done
- Phase 0 audit; Phase 1 leakage-safe pipeline + time folds; Phase 2 model comparison (LR chosen).
- Phase 3: weighting decision (x5 post-May, mechanical checks, all passed), final LR fit on 11,142 labelled claims, predictions.csv (2,252 rows, validated), expected-score range, rupee impact, evidence report, memo to Ritu, README_PHASE3, 11 new tests.

## Final model
Logistic regression C=0.5, 26 features, post-May rows weighted x5 (regime adaptation, one-fold evidence; x20 not used). Bundle: artifacts/phase3/final_model.joblib (loader kestrel/final.py: score(), contributions(), reasons()).

## Expected hidden score
PR-AUC about 0.28 (plausible 0.17-0.52; near chance ~0.03 if new fraud outlets appear). Accuracy not meaningful (do-nothing 97-99%).

## Rupee impact (3 months, 120 checks, extrapolated from validation)
Central net +Rs 19.1k (+159/check, +6.4k/month; -12.1k if Rs 260 contact cost applies); optimistic +34.6k; pessimistic -42.1k.
Only 92 of 2,252 test claims have positive expected value (67 with contact cost): review fewer than 120 if needed.

## Files (Phase 3)
kestrel/final.py, scripts/run_phase3.py, scripts/validate_submission.py, tests/test_phase3.py, predictions.csv, README_PHASE3.md,
docs/PHASE3_EVIDENCE_REPORT.md, docs/MEMO_TO_RITU.md, artifacts/phase3/*; changed: kestrel/models.py (feature-name support only), requirements.txt, .gitignore unchanged.

## Assumptions
LAG_DAYS=14, EB strength 20, Rs 260 contact cost = sensitivity only, undecided cases unused, central score = equal-weighted mean of 3 staleness scenarios.

## Risks
New fraud outlets with no history; stale partner history; cold-start partners (81 test claims); x5 chosen on same fold as evaluated; thin economics if contact cost applies; 7 outlets drive the queue; threshold-gaming signal (Rs 1,950-1,999) not modelled.

## Phase 4 (service + UI)
Files: app.py, static/index.html, kestrel/service.py, tests/synthetic_bundle.py, tests/test_phase4_service.py, tests/test_phase4_api.py, README.md, requirements.txt (pins), requirements-dev.txt, .gitignore (+venv, .env). Phase 1-3 code, model bundle and predictions.csv unchanged.
Risk levels: high >= 0.13 (~top 5% of test scores), medium >= 0.04, else low. Unseen outlet = scored as no-history outlet (+warning). Unknown SKU -> 422.
Known wording issue (Phase 3 `final.py::_text`, left unchanged): the outlet-history reason can read "0 confirmed fraud among N decided claims" when thin/no history is what raises the score.
Phase 4 verification still to do (needs internet): fresh venv, `pip install -r requirements-dev.txt`, `python -m unittest discover -s tests -t .` (HTTP tests should no longer skip), `uvicorn app:app --reload`, curl /health + /predict (valid, unseen outlet, bad input), open UI at http://127.0.0.1:8000/. fastapi/uvicorn are range-pinned, not exact.

## Remaining deliverables (later phases)
Phase 5: submission-form.md (incl. expected score text from evidence report s6), AI disclosure/cost, demo script/recording, final QA.
