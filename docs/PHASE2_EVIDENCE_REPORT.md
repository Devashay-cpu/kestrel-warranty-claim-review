# Phase 2 - validation evidence report

Data: 11,142 labelled unique claims (141 fraud), Phase 1 pipeline unchanged. Folds (time-based, 14-day label embargo, partner history frozen at fold cutoff): A Feb-Apr (23 fraud / 2,141), **B May-Jun headline (36 / 1,422)**, C June (22 / 713). K = 40 claims per month of window (A 120, B 80, C 40). Hold-out file never read; no random splits. Full tables: `PHASE2_RESULTS_TABLES.md`.

## 1. Accuracy is not evidence
Predicting "no fraud" for everything scores 98.9% (A), 97.5% (B), 96.9% (C) accuracy. The board's >97% KPI is met by doing nothing in A and B and missed by doing nothing in C. Flagging 40-80 claims lowers accuracy (LR at capacity: 94.7%, 92.0%, 94.1%) because most flags are genuine claims. Accuracy cannot rank models here; PR-AUC and rupees can.

## 2. Ranking (PR-AUC, 95% partner-cluster bootstrap)
| Fold | prevalence | rule prior-claims | rule partner-rate | LR | HGB | RF |
|---|---|---|---|---|---|---|
| A | 0.011 | 0.089 | 0.174 | 0.598 | 0.604 | 0.606 |
| **B** | 0.025 | 0.033 | 0.042 | 0.023 [0.010,0.042] | 0.020 [0.009,0.036] | 0.023 [0.009,0.042] |
| C | 0.031 | 0.048 | 0.123 | 0.365 [0.024,0.655] (bootstrap mean 0.368) | 0.263 [0.077,0.449] | 0.248 [0.042,0.502] |
Intervals are wide (22-36 positives, fraud clustered at 8-14 outlets). Fold A CIs for LR: [0.205, 0.827].

## 3. Capacity view (LR, top-K by probability; Rs per checked claim, goodwill Rs 380 per genuine claim held)
| Fold | K | frauds caught | precision | recall | fraud Rs stopped / check | net / check (after goodwill) | net after Rs 260 contact | rule prior-claims net/check | best possible (oracle) Rs/check |
|---|---|---|---|---|---|---|---|---|---|
| A | 120 | 15 | 12.5% | 65.2% | 986 | +654 | +394 | +113 | 1,193 |
| **B** | 80 | 1 | 1.2% | 2.8% | 25 | -351 | -611 | -259 | 839 |
| C | 40 | 10 | 25.0% | 45.5% | 406 | +121 | -139 | -212 | 1,142 |
Random checking would stop about Rs 47-67 per claim. Bootstrap CI for LR top-K net/check on C: [-380, +665]; on B: [-380, -292].
Ranking by expected value (p x amount) gave identical top-K in all folds, so amount-aware ranking adds nothing yet.

## 4. Where it fails (headline fold B, `failure_cases_headline_B.csv`)
79 of 80 flags were genuine claims (45 different partners, 97% of them onboarded before April 2025, 62% auto-approved small claims, median Rs 1,995, mean 0.73 prior claims per customer); 35 of 36 frauds were missed. Missed frauds: 97% are auto-approved small claims (median Rs 1,406), 80% sit at five outlets (SP3160, SP3232, SP3318, SP3129, SP3319) with no fraud history at the freeze date. In fold C, 9 of 22 frauds were at unseen partners; PR-AUC fell from 0.50 (1st half of June) to 0.18 (2nd half), i.e. the partner history goes stale within weeks.

## 5. Regime / uninspected claims
Restricted to auto-approved (small, uninspected) claims, which are 77% of both B and the hold-out: LR PR-AUC 0.030 in B (prevalence 0.032) and 0.388 in C (prevalence 0.038), the same picture as overall. Inspection features are not just noise: dropping them costs -0.04 on A.

## 6. Limits
~36 post-May positives; folds B and C overlap in time (June); hyper-parameters and the family were chosen on A / B+C, so B and C numbers carry mild selection optimism; LAG_DAYS=14, EB strength 20 and the Rs 260 contact-cost reading are assumptions; undecided cases excluded; a model trained now sees post-May labels that fold B never had, so B understates and C (1 month horizon) overstates what a 3-month hold-out will look like. Expect something in between and say so as a range, not a point, in Phase 3.
