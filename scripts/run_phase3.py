"""Phase 3: weighting decision -> final fit -> predictions.csv -> expected score range -> rupee impact.
Run from project root:  python scripts/run_phase3.py        (needs Phase 2 artifacts: artifacts/phase2/model_selection.json)
Test labels do not exist; the unlabelled file is only SCORED at the end, never used to choose anything.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib, numpy as np, pandas as pd
from scipy.stats import spearmanr
from kestrel import config as C, features as F, splits as S, pipeline as P, models as M, evaluate as E, final as FIN
from kestrel.clean import load_all
from validate_submission import validate

OUT = C.ART / "phase3"; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time(); SEED = 0; REPS = 1000
sel = json.load(open(C.ART / "phase2" / "model_selection.json"))
assert sel["chosen_family"] == "lr"
FEATS = sel["final_features"]; PARAMS = sel["configs"]["lr"]; assert FEATS == F.feature_names()
ds = load_all(); ms = ds["model_set"]; Xtr = P.train_features(ds)
AMT = ms.set_index("claim_id").claim_amount_inr
W5 = lambda df: np.where(df.regime_post_may.values == 1, 5.0, 1.0)

# ---------------- 1. scenarios: B headline, C fresh, C with staler partner history ----------------
def scenario(fold_name, stale_days):
    fold = next(f for f in S.FOLDS if f["name"] == fold_name)
    tr_m, va_m = S.fold_masks(ms, fold)
    frz = S.fold_cutoff(fold) - pd.Timedelta(days=stale_days)
    ev = P.events_of(ms[ms.submitted_at <= frz])
    va = F.build_features(ms[va_m], ds["partners"], ds["products"], ds["log"], ev, ("frozen", frz))
    tr = Xtr[tr_m.values].reset_index(drop=True)
    k = {"B_test_like": 80, "C_june": 40}[fold_name]
    rng = np.random.default_rng(1000 + [f["name"] for f in S.FOLDS].index(fold_name))
    return dict(tr=tr, va=va, y=va.is_fraud.values, amt=AMT.loc[va.claim_id].values, jit=rng.random(len(va)), k=k, part=va.partner_id.values)

SC_DEF = [("B_headline", "B_test_like", 0), ("C_fresh", "C_june", 0), ("C_stale30", "C_june", 30), ("C_stale60", "C_june", 60)]
rows, boots, SC, EV, halves = [], [], {}, {}, []
for name, fn, stale in SC_DEF:
    d = scenario(fn, stale); sc, ev_ = {"rule_prior": d["va"].customer_prior_claims.values.astype(float)}, {}
    for wname, w in [("uniform", None), ("x5", W5(d["tr"]))]:
        p, _ = M.fit_predict("lr", PARAMS, FEATS, d["tr"], d["va"], weights=w, seed=SEED)
        sc[wname] = p; ev_[wname] = E.expected_value(p, d["amt"])
        m = E.core_metrics(d["y"], d["amt"], p, d["jit"], d["k"], ev_[wname])
        th = E.threshold_stats(d["y"], d["amt"], p)
        rows.append(dict(scenario=name, weighting=wname, n_val=len(d["y"]), n_fraud=int(d["y"].sum()), k=d["k"], **m,
                         calib_mean_pred_over_prev=p.mean() / d["y"].mean(), ev_rule_flagged=th["n_flagged"], ev_rule_net_inr=th["net_inr"]))
        if fn == "C_june" and stale == 0:
            t = pd.to_datetime(d["va"].submitted_at).values; m1 = t < np.datetime64("2026-06-16")
            for h, mm in (("first_15d", m1), ("after_15d", ~m1)):
                halves.append(dict(weighting=wname, window=h, n_fraud=int(d["y"][mm].sum()), pr_auc=E.average_precision_score(d["y"][mm], p[mm])))
    SC[name], EV[name] = sc, ev_
    boot = E.cluster_bootstrap(d["y"], d["amt"], d["part"], d["jit"], d["k"], sc, ev_, reps=REPS, seed=42)
    for wname in ("uniform", "x5"):
        for met in ("pr_auc", "k_precision", "k_stopped_per_check", "k_net_per_check"):
            v = boot[(wname, met)]; boots.append(dict(scenario=name, weighting=wname, metric=met, mean=v.mean(), lo=v.quantile(.025), hi=v.quantile(.975)))
    for met in ("pr_auc", "k_precision", "k_net_per_check"):
        dlt = boot[("x5", met)] - boot[("uniform", met)]
        boots.append(dict(scenario=name, weighting="x5_minus_uniform", metric=met, mean=dlt.mean(), lo=dlt.quantile(.025), hi=dlt.quantile(.975), p_x5_better=float((dlt > 0).mean())))
    print(name, "done", f"{time.time()-t0:.0f}s")
R = pd.DataFrame(rows); R.to_csv(OUT / "weighting_comparison_scenarios.csv", index=False)
BT = pd.DataFrame(boots); BT.to_csv(OUT / "weighting_bootstrap_ci.csv", index=False)
pd.DataFrame(halves).to_csv(OUT / "weighting_halves_foldC.csv", index=False)

# ---------------- 2. weighting decision (rule written down here, applied mechanically) ----------------
g = lambda s, w, c: float(R[(R.scenario == s) & (R.weighting == w)][c].iloc[0])
hv = pd.DataFrame(halves).pivot(index="window", columns="weighting", values="pr_auc")
pbetter = float(BT[(BT.scenario == "C_fresh") & (BT.weighting == "x5_minus_uniform") & (BT.metric == "pr_auc")].p_x5_better.iloc[0])
checks = {
    "paired_bootstrap_P(x5>uniform PR-AUC)_foldC_>=0.80": pbetter >= 0.80,
    "x5>=uniform PR-AUC in both halves of June": bool((hv["x5"] >= hv["uniform"]).all()),
    "x5>=uniform PR-AUC in C_fresh, C_stale30, C_stale60 (>=2 of 3 required)": sum(g(s, "x5", "pr_auc") >= g(s, "uniform", "pr_auc") for s in ("C_fresh", "C_stale30", "C_stale60")) >= 2,
    "x5 top-40 net/check not worse than uniform (C_fresh)": g("C_fresh", "x5", "k_net_per_check") >= g("C_fresh", "uniform", "k_net_per_check"),
    "x5 calibration ratio <= 2.0 on C_fresh (not wildly over-confident)": g("C_fresh", "x5", "calib_mean_pred_over_prev") <= 2.0,
    "x5 no worse than uniform on B (no post-May rows -> must be identical)": abs(g("B_headline", "x5", "pr_auc") - g("B_headline", "uniform", "pr_auc")) < 1e-9,
}
USE_X5 = all(checks.values()); WEIGHTING = "x5" if USE_X5 else "uniform"
print("weighting checks:", checks, "->", WEIGHTING)

# ---------------- 3. final fit on ALL eligible labelled rows ----------------
w_final = W5(Xtr) if USE_X5 else None
_, pipe = M.fit_predict("lr", PARAMS, FEATS, Xtr, Xtr.iloc[:5], weights=w_final, seed=SEED)
_, pipe_u = M.fit_predict("lr", PARAMS, FEATS, Xtr, Xtr.iloc[:5], weights=None, seed=SEED)
pre = pipe.named_steps["pre"]; Z = FIN._dense(pre.transform(Xtr[FEATS]))
names = list(pre.get_feature_names_out()); cats = [f for f in FEATS if f in F.CAT_FEATURES]
def src(n):
    n = n.split("__", 1)[1]
    return n if n in FEATS else next(c for c in cats if n.startswith(c + "_"))
bundle = dict(pipeline=pipe, features=FEATS, params=PARAMS, weighting=WEIGHTING, baseline=Z.mean(0), out_to_src=[src(n) for n in names],
              label_cutoff=ms.submitted_at.max(), events=P.events_of(ms), log=ds["log"], partners=ds["partners"], products=ds["products"],
              constants=dict(LAG_DAYS=C.LAG_DAYS, WINDOW_DAYS=C.WINDOW_DAYS, EB_STRENGTH=C.EB_STRENGTH, GOODWILL_INR=C.GOODWILL_INR, CONTACT_INR=C.CONTACT_INR,
                             SMALL_CLAIM_INR=C.SMALL_CLAIM_INR, REGIME_DATE=str(C.REGIME_DATE.date())),
              train_rows=len(Xtr), train_fraud=int(Xtr.is_fraud.sum()), train_range=[str(ms.submitted_at.min()), str(ms.submitted_at.max())])
joblib.dump(bundle, OUT / "final_model.joblib")

coef = pd.DataFrame({"feature": names, "coef": pipe.named_steps["clf"].coef_[0], "coef_uniform_fit": pipe_u.named_steps["clf"].coef_[0]})
coef["abs"] = coef.coef.abs(); coef.sort_values("abs", ascending=False).drop(columns="abs").to_csv(OUT / "final_model_coefficients.csv", index=False)
flips = int(((np.sign(coef.coef) != np.sign(coef.coef_uniform_fit)) & (coef.coef.abs() > 0.05) & (coef.coef_uniform_fit.abs() > 0.05)).sum())

# ---------------- 4. predictions.csv ----------------
te_raw = pd.read_csv(C.RAW / "test_unlabelled.csv"); sample = pd.read_csv(C.RAW / "sample_submission.csv")
b2 = FIN.load_bundle(OUT / "final_model.joblib")
Xte, p_te = FIN.score(te_raw, b2)
Xte_phase1 = P.test_features(ds)                       # must equal the bundle path (same pipeline, no drift)
assert np.allclose(Xte_phase1.set_index("claim_id").loc[Xte.claim_id, FEATS].select_dtypes("number").values, Xte[FEATS].select_dtypes("number").values)
pred = pd.DataFrame({"claim_id": Xte.claim_id.values, "score": p_te}).set_index("claim_id").loc[sample.claim_id].reset_index()
pred["score"] = pred.score.round(8)
pred.to_csv(C.ROOT / "predictions.csv", index=False)
errs = validate(C.ROOT / "predictions.csv"); assert not errs, errs
X_sorted = Xte.set_index("claim_id").loc[sample.claim_id].reset_index(); X_sorted["score"] = pred.score.values

# sensitivity: uniform vs x5 on the test claims (no labels used)
p_u = pipe_u.predict_proba(X_sorted[FEATS])[:, 1]
sens = dict(spearman_x5_vs_uniform=float(spearmanr(X_sorted.score, p_u)[0]),
            top120_overlap_x5_vs_uniform=int(len(set(X_sorted.nlargest(120, "score").claim_id) & set(X_sorted.assign(u=p_u).nlargest(120, "u").claim_id))))

# ---------------- 5. test prediction summary, monthly top-40 queue with reasons ----------------
X_sorted["amount"] = np.expm1(X_sorted.log_amount); X_sorted["month"] = pd.to_datetime(X_sorted.submitted_at).dt.to_period("M").astype(str)
X_sorted["ev"] = E.expected_value(X_sorted.score.values, X_sorted.amount.values)
top = (X_sorted.sort_values(["month", "score"], ascending=[True, False]).groupby("month").head(C.CAPACITY_PER_MONTH).copy())
top["month_rank"] = top.groupby("month").score.rank(ascending=False, method="first").astype(int)
rs = FIN.reasons(b2, top, top=3)
for i in range(3): top[f"reason_{i+1}"] = [r[i]["text"] if len(r) > i else "" for r in rs]
top[["month", "month_rank", "claim_id", "partner_id", "submitted_at", "amount", "score", "customer_prior_claims", "auto_approved", "p_lab_fraud", "p_lab_n",
     "p_claims_30d", "reason_1", "reason_2", "reason_3"]].to_csv(OUT / "test_top_claims.csv", index=False)
cold = X_sorted.p_lab_n == 0
summ = {"n_test_claims": len(X_sorted), "score_mean": X_sorted.score.mean(), "score_median": X_sorted.score.median(), "score_p90": X_sorted.score.quantile(.9),
        "score_p99": X_sorted.score.quantile(.99), "score_max": X_sorted.score.max(), "score_min": X_sorted.score.min(),
        "sum_scores_expected_frauds_if_calibrated": X_sorted.score.sum(), "share_auto_approved_small": X_sorted.auto_approved.mean(),
        "claims_at_partners_with_no_decided_history": int(cold.sum()), "partners_in_test": int(X_sorted.partner_id.nunique()),
        "queue_size_3_months": len(top), "queue_distinct_partners": int(top.partner_id.nunique()),
        "queue_top5_partner_share": float(top.partner_id.value_counts().head(5).sum() / len(top)),
        "queue_share_auto_approved": float(top.auto_approved.mean()), "queue_median_amount": float(top.amount.median()),
        "queue_share_partner_with_known_fraud": float((top.p_lab_fraud > 0).mean()),
        "claims_with_positive_expected_value": int((X_sorted.ev > 0).sum()), "claims_with_ev_above_contact_cost": int((X_sorted.ev > C.CONTACT_INR).sum()),
        **sens}
pd.DataFrame([{"metric": k, "value": v} for k, v in summ.items()]).to_csv(OUT / "test_prediction_summary.csv", index=False)
top5 = top.partner_id.value_counts().head(8).rename_axis("partner_id").reset_index(name="queue_claims"); top5.to_csv(OUT / "test_queue_partner_concentration.csv", index=False)

# ---------------- 6. expected hidden score range (from actual validation runs of the FINAL weighting) ----------------
fw = WEIGHTING
pr = {s: g(s, fw, "pr_auc") for s, _, _ in SC_DEF}; prev = {s: g(s, fw, "prevalence") for s, _, _ in SC_DEF}
ci = {s: BT[(BT.scenario == s) & (BT.weighting == fw) & (BT.metric == "pr_auc")][["lo", "hi"]].iloc[0].tolist() for s, _, _ in SC_DEF}
central = float(np.mean([pr["C_fresh"], pr["C_stale30"], pr["C_stale60"]]))
exp = dict(metric="PR-AUC (average precision) over all test claims", weighting=fw,
           validation_pr_auc=pr, validation_prevalence=prev, validation_pr_auc_ci95=ci,
           low=pr["B_headline"], likely_low=min(pr["C_stale30"], pr["C_stale60"]), central=central, high=pr["C_fresh"],
           method="low = fold B (new fraud outlets with no history -> chance level); high = fold C fresh history; likely_low = fold C with history 30-60 days older; "
                  "central = mean of C_fresh, C_stale30, C_stale60, i.e. the three staleness bands (mean 28/58/88 days) weighted equally, matching a hidden window whose claims "
                  "are 1-92 days after the 30 Jun freeze. x5 weighting was chosen on this same fold, so treat C numbers as mildly optimistic.",
           staleness_days={"C_fresh": "14-43", "C_stale30": "44-73", "C_stale60": "74-103", "B_headline": "14-74 (and no post-May training rows)", "test": "1-92"},
           test_prevalence_unknown=True, comparator_prevalence_range=[0.013, 0.031],
           accuracy_note="all-negative accuracy would be ~97-98.7% depending on the (unknown) fraud rate; accuracy is not the metric")
json.dump(exp, open(OUT / "expected_score_range.json", "w"), indent=2, default=float)

# ---------------- 7. rupee impact for the 3-month test window (extrapolated from validation, NOT test outcomes) ----------------
K = C.CAPACITY_PER_MONTH * 3; imp = []
def scen_row(label, s):
    r = R[(R.scenario == s) & (R.weighting == fw)].iloc[0]
    return r["k_precision"], r["k_stopped_per_check"]
sc_pts = {"pessimistic (fold B: new outlets, no history)": scen_row("", "B_headline")}
cm = [scen_row("", x) for x in ("C_fresh", "C_stale30", "C_stale60")]
sc_pts["central (mean of fold C fresh / 30d-stale / 60d-stale)"] = (float(np.mean([c[0] for c in cm])), float(np.mean([c[1] for c in cm])))
sc_pts["optimistic (fold C: fresh history)"] = cm[0]
for lab, (prec, stp) in sc_pts.items():
    tp = prec * K; fp = K - tp; stopped = stp * K; gw = C.GOODWILL_INR * fp; net = stopped - gw
    imp.append(dict(scenario=lab, test_claims=len(X_sorted), reviewed_3_months=K, reviewed_per_month=C.CAPACITY_PER_MONTH, share_of_claims_reviewed=K / len(X_sorted),
                    expected_frauds_caught=tp, expected_precision=prec, fraud_inr_stopped=stopped, goodwill_cost_inr=gw, net_benefit_inr=net,
                    stopped_per_check_inr=stp, net_per_check_inr=net / K, net_after_Rs260_contact_inr=net - C.CONTACT_INR * K,
                    net_per_check_after_contact_inr=net / K - C.CONTACT_INR, net_per_month_inr=net / 3, net_per_month_after_contact_inr=(net - C.CONTACT_INR * K) / 3,
                    stopped_per_month_inr=stopped / 3))
IMP = pd.DataFrame(imp); IMP.to_csv(OUT / "business_impact.csv", index=False)

# ---------------- 8. metrics + summary ----------------
R.to_csv(OUT / "final_model_metrics.csv", index=False)
json.dump(dict(model="LogisticRegression (L2) in sklearn Pipeline", params=PARAMS, features=FEATS, n_features=len(FEATS), weighting=WEIGHTING,
               weighting_checks={k: bool(v) for k, v in checks.items()}, p_x5_better_foldC=pbetter, coefficient_sign_flips_x5_vs_uniform=flips,
               train_rows=len(Xtr), train_fraud=int(Xtr.is_fraud.sum()), train_post_may_rows=int(Xtr.regime_post_may.sum()),
               train_post_may_fraud=int(Xtr[Xtr.regime_post_may == 1].is_fraud.sum()), label_cutoff=str(ms.submitted_at.max()),
               test_rows=len(pred), excluded=dict(undecided=len(ds["undecided"]), injected_text=len(ds["excluded"])), bundle="artifacts/phase3/final_model.joblib",
               effective_post_may_share_after_x5=float(Xtr.regime_post_may.mean() * 5 / (Xtr.regime_post_may.mean() * 5 + 1 - Xtr.regime_post_may.mean())) if USE_X5 else float(Xtr.regime_post_may.mean()),
               x20_not_used="no evidence-based reason; x5 chosen conservatively"), open(OUT / "final_model_summary.json", "w"), indent=2, default=float)
print(R[["scenario", "weighting", "pr_auc", "prevalence", "k_precision", "k_stopped_per_check", "k_net_per_check", "calib_mean_pred_over_prev"]].round(3).to_string())
print(BT[BT.weighting == "x5_minus_uniform"].round(3).to_string()); print(hv)
print(json.dumps(exp, indent=1, default=float)); print(IMP.round(1).T.to_string()); print(json.dumps(summ, indent=1, default=float))
print(f"finished {time.time()-t0:.0f}s")
