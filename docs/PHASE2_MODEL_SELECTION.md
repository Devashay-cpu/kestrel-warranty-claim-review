# Phase 2 - model-selection decision

**Chosen family: regularised Logistic Regression (C = 0.5), 26 default features, no optional groups.**
Decision procedure was fixed in `scripts/run_phase2.py` before looking at results: hyper-parameters chosen on fold A only;
family = simplest (LR < HGB < RF) whose mean PR-AUC over folds B and C is >= 90% of the best family AND beats the best rule baseline.

| Model | PR-AUC A | PR-AUC B (headline) | PR-AUC C | mean(B,C) |
|---|---|---|---|---|
| prevalence (no skill) | 0.011 | 0.025 | 0.031 | |
| rule: customer_prior_claims | 0.089 | 0.033 | 0.048 | 0.041 |
| rule: partner fraud rate | 0.174 | 0.042 | 0.123 | 0.083 |
| **LR** | 0.598 | 0.023 | 0.365 | **0.194** |
| HGB | 0.604 | 0.020 | 0.263 | 0.141 |
| RF | 0.606 | 0.023 | 0.248 | 0.135 |

## Honest reading
1. **On the headline fold B no model has measurable skill.** PR-AUC 0.020-0.023 is at or below prevalence (0.025); the rule baselines are slightly better (0.033-0.042). Cause (diagnostic, `diagnostic_unseen_fraud_partners.csv`): 100% of the 36 fraud cases in B sit at partners with no known fraud at the freeze date, and 5 outlets hold 81% of them. The fraud wave that followed the May change is invisible to a model trained on pre-May data. Label-free volume features did not rescue it.
2. **On fold C (June, trained with ~380 post-May rows)** LR reaches PR-AUC 0.365 (11.8x prevalence); paired bootstrap vs the prior-claims rule: +0.32 [0.003, 0.61], P(better)=0.986. HGB/RF are lower (0.26/0.25) but their CIs overlap LR; differences between the three families are NOT statistically established. On fold A all three are equal (0.60).
3. So the evidence supports: partner label history is what carries the signal (removing it costs -0.18 to -0.32 PR-AUC on A and C across the three families), and it decays with time since the freeze (June, first half 0.50 -> second half 0.18). It does not support any claim of stable skill under a regime with no post-May labels.

## Why LR over HGB/RF
Equal on A, better on C, fewer parameters, deterministic (seed has no effect: PR-AUC identical across 3 seeds), coefficients are readable for the "reasons" required later, and least overfit-prone with 141 positives. The pre-specified rule selects it; LR was also the best on mean(B,C), so the simplicity tie-break was not needed. Seed stability was checked for LR only.

## Optional feature groups (rule: adopt additions only if dPR-AUC >= +0.01 on B and C and >= -0.005 on A)
| Group | dA | dB | dC | Decision |
|---|---|---|---|---|
| partner_age | +0.014 | -0.007 | -0.015 | rejected (hurts B and C; also drifts with calendar time) |
| serial (format, lower-case, reuse) | -0.004 | -0.001 | -0.029 | rejected |
| desc (12 categories) | -0.005 | -0.001 | -0.020 | rejected |
| drop inspection features | -0.039 | -0.001 | -0.012 | rejected (not harmless) |
(LR rows. Other families in `ablation_delta_vs_base.csv`: no consistent gain either.)

## Regime handling (exploratory, NOT adopted into the Phase 2 numbers)
Up-weighting post-May rows on fold C lifts LR PR-AUC 0.365 -> 0.516 (x5) -> 0.578 (x20), and the gain holds in both halves of June. Only one fold (22 positives) can test it, B has no post-May training rows, so this is not validated enough to be a Phase 2 result. **Recommendation for Phase 3: fit the final model with post-May weight x5 (conservative; final fit has ~1,420 post-May rows vs 380 in fold C), and say it was chosen on one fold.**

## Decision threshold vs ranking (kept separate)
Ranking quality = PR-AUC / top-K. Operational rule = check a claim only if p*amount - (1-p)*380 > 0 (needs calibrated p; LR mean-pred / prevalence = 1.31, 0.50, 1.38 on A, B, C, i.e. not stable under the regime shift). Per-claim break-even precision is 380/(amount+380): a Rs 800 claim needs p > 32% to be worth checking. In C the rule flags 45 claims (capacity 40): precision 22%, Rs 16,233 stopped, net +Rs 2,933 after goodwill; 26 claims if the Rs 260 contact cost is also charged (net +Rs 1,587). In B it flags 34 claims with 0 frauds (net -Rs 12,920).
