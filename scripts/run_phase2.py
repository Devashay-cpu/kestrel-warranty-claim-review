"""Phase 2: honest out-of-time model comparison. Run from project root:  python scripts/run_phase2.py
Never reads the unlabelled hold-out file or its features. Hyper-parameters are chosen on fold A only; B (May-Jun) is the headline.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from kestrel import config as C, features as F, splits as S, pipeline as P, models as M, evaluate as E
from kestrel.clean import load_all

OUT = C.ART / "phase2"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 0; REPS = 1000
t0 = time.time()
ds = load_all(); ms = ds["model_set"]
Xtr = P.train_features(ds)
AMT = ms.set_index("claim_id").claim_amount_inr
BASE = F.feature_names()
INSPECTION = ["partner_inspected", "has_note", "note_cat"]
SETS = {"base": BASE,
        "+partner_age": BASE + F.GROUPS["partner_age"], "+serial": BASE + F.GROUPS["serial"], "+desc": BASE + F.GROUPS["desc"],
        "-inspection": [f for f in BASE if f not in INSPECTION],
        "-partner_label(info only)": [f for f in BASE if f not in F.GROUPS["partner_label"]]}
FOLD_K = {"A_pre_regime": 120, "B_test_like": 80, "C_june": 40}     # literal capacity: 40 claims/month x months in window
FAMS = M.SIMPLICITY_ORDER

# ---- fold data (val history frozen at fold cutoff; train rows rolling point-in-time) ----
FD = {}
for fold in S.FOLDS:
    tr, va = P.fold_data(ds, fold, Xtr)
    rng = np.random.default_rng(1000 + len(FD))
    FD[fold["name"]] = dict(tr=tr, va=va, y=va.is_fraud.values, amt=AMT.loc[va.claim_id].values,
                            jit=rng.random(len(va)), k=FOLD_K[fold["name"]], part=va.partner_id.values)
print(f"data ready {time.time()-t0:.0f}s", {k: (len(v['tr']), len(v['va']), int(v['y'].sum())) for k, v in FD.items()})

def rule_scores(va):
    return {"rule_prior": va.customer_prior_claims.values.astype(float),   # simple risk rule: repeat claimants first
            "rule_partner": va.p_rate_eb.values}                           # partner point-in-time fraud rate only

def metrics_row(name, fold, score, ev=None, extra=None):
    d = FD[fold]
    r = dict(model=name, fold=fold, n_val=len(d["y"]), n_fraud=int(d["y"].sum()))
    r.update(E.core_metrics(d["y"], d["amt"], score, d["jit"], d["k"], ev))
    r["accuracy_all_negative"] = 1 - r["prevalence"]
    aa = d["va"].auto_approved.values == 1
    if aa.sum() > 0 and d["y"][aa].sum() > 0:
        r["aa_n"] = int(aa.sum()); r["aa_fraud"] = int(d["y"][aa].sum())
        r["aa_pr_auc"] = E.average_precision_score(d["y"][aa], score[aa]); r["aa_roc_auc"] = E.roc_auc_score(d["y"][aa], score[aa])
    r["oracle_stopped_per_check"] = E.oracle_per_check(d["y"], d["amt"], d["k"])
    r["random_stopped_per_check"] = float((d["y"] * d["amt"]).sum() / len(d["y"]))
    if ev is not None:   # probability-quality check (calibration matters only for the EV decision rule)
        r["brier"] = E.brier_score_loss(d["y"], score); r["mean_pred_over_prev"] = score.mean() / d["y"].mean()
    if extra: r.update(extra)
    return r

# ---- 1. hyper-parameter selection on fold A only ----
grid_rows, SEL = [], {}
for kind in FAMS:
    for prm in M.GRIDS[kind]:
        for fn, d in FD.items():
            p, _ = M.fit_predict(kind, prm, BASE, d["tr"], d["va"], seed=SEED)
            grid_rows.append(metrics_row(f"{kind}:{prm['name']}", fn, p, ev=E.expected_value(p, d["amt"])))
grid = pd.DataFrame(grid_rows); grid.to_csv(OUT / "grid_all_folds.csv", index=False)
for kind in FAMS:
    a = grid[(grid.fold == "A_pre_regime") & grid.model.str.startswith(kind + ":")].sort_values("pr_auc", ascending=False)
    best = a.iloc[0].model.split(":")[1]
    SEL[kind] = next(p for p in M.GRIDS[kind] if p["name"] == best)
print("selected on fold A:", {k: v["name"] for k, v in SEL.items()}, f"{time.time()-t0:.0f}s")

# ---- 2. main comparison, base features ----
SC, EV = {f: {} for f in FD}, {f: {} for f in FD}
rows = []
for fn, d in FD.items():
    for name, s in rule_scores(d["va"]).items():
        SC[fn][name] = s; rows.append(metrics_row(name, fn, s))
    for kind in FAMS:
        p, _ = M.fit_predict(kind, SEL[kind], BASE, d["tr"], d["va"], seed=SEED)
        SC[fn][kind] = p; EV[fn][kind] = E.expected_value(p, d["amt"])
        rows.append(metrics_row(kind, fn, p, ev=EV[fn][kind], extra=dict(config=SEL[kind]["name"])))
comp = pd.DataFrame(rows); comp.to_csv(OUT / "model_comparison_by_fold.csv", index=False)

# ---- 3. pre-specified selection rule ----
def mean_bc(name): return comp[(comp.model == name) & comp.fold.isin(["B_test_like", "C_june"])].pr_auc.mean()
best_model = max(FAMS, key=mean_bc); best_rule = max(["rule_prior", "rule_partner"], key=mean_bc)
cands = [f for f in FAMS if mean_bc(f) >= 0.90 * mean_bc(best_model)]
chosen = next(f for f in FAMS if f in cands)       # simplest within 90% of best (order lr < hgb < rf)
if mean_bc(chosen) <= mean_bc(best_rule): chosen = "RULE"
print("mean PR-AUC (B,C):", {n: round(mean_bc(n), 4) for n in ["rule_prior", "rule_partner"] + FAMS}, "-> family", chosen)

# ---- 4. feature-group ablation (all families, selected configs) ----
arows = []
for kind in FAMS:
    for sname, feats in SETS.items():
        for fn, d in FD.items():
            p, _ = M.fit_predict(kind, SEL[kind], feats, d["tr"], d["va"], seed=SEED)
            arows.append(dict(family=kind, feature_set=sname, fold=fn, n_features=len(feats), pr_auc=E.average_precision_score(d["y"], p),
                              roc_auc=E.roc_auc_score(d["y"], p), **{f"k_{a}": b for a, b in E.topk_stats(d["y"], d["amt"], p, d["jit"], d["k"]).items() if a in ("precision", "recall", "stopped_per_check")}))
abl = pd.DataFrame(arows); abl.to_csv(OUT / "ablation_feature_groups.csv", index=False)
piv = abl.pivot_table(index=["family", "feature_set"], columns="fold", values="pr_auc")
delta = piv.sub(piv.xs("base", level="feature_set").reindex(piv.index.get_level_values(0)).values)
delta.columns = [f"d_{c}" for c in delta.columns]; delta.to_csv(OUT / "ablation_delta_vs_base.csv")

decisions = {}
fam_for_rule = chosen if chosen != "RULE" else "lr"
dl = delta.xs(fam_for_rule, level="family")
for sname in ["+partner_age", "+serial", "+desc"]:
    r = dl.loc[sname]; ok = r.d_B_test_like >= 0.01 and r.d_C_june >= 0.01 and r.d_A_pre_regime >= -0.005
    decisions[sname] = dict(adopt=bool(ok), **{c: round(float(v), 4) for c, v in r.items()})
r = dl.loc["-inspection"]; ok = r.d_B_test_like >= 0 and r.d_C_june >= 0 and r.d_A_pre_regime >= -0.005
decisions["-inspection"] = dict(adopt=bool(ok), **{c: round(float(v), 4) for c, v in r.items()})
final_feats = [f for f in BASE]
for s in ["+partner_age", "+serial", "+desc"]:
    if decisions[s]["adopt"]: final_feats += [f for f in SETS[s] if f not in final_feats]
if decisions["-inspection"]["adopt"]: final_feats = [f for f in final_feats if f not in INSPECTION]
print("feature decisions:", {k: v["adopt"] for k, v in decisions.items()}, "final n =", len(final_feats))

# ---- 5. regime: recency weighting on fold C (only fold with post-May training rows) ----
wrows = []
d = FD["C_june"]
for kind in FAMS:
    for wname, w in [("uniform", None), ("post_may_x5", np.where(d["tr"].regime_post_may == 1, 5.0, 1.0)), ("post_may_x20", np.where(d["tr"].regime_post_may == 1, 20.0, 1.0))]:
        p, _ = M.fit_predict(kind, SEL[kind], final_feats, d["tr"], d["va"], weights=w, seed=SEED)
        wrows.append(dict(family=kind, weighting=wname, n_post_may_train=int(d["tr"].regime_post_may.sum()), pr_auc=E.average_precision_score(d["y"], p),
                          k_precision=E.topk_stats(d["y"], d["amt"], p, d["jit"], d["k"])["precision"]))
pd.DataFrame(wrows).to_csv(OUT / "regime_weighting_foldC.csv", index=False)

# ---- 6. final model rows (chosen family, final features) + seed stability ----
FINAL = {}
frows, srows = [], []
for fn, d in FD.items():
    if chosen == "RULE": break
    p, m = M.fit_predict(chosen, SEL[chosen], final_feats, d["tr"], d["va"], seed=SEED)
    FINAL[fn] = p; SC[fn]["final"] = p; EV[fn]["final"] = E.expected_value(p, d["amt"])
    frows.append(metrics_row(f"final_{chosen}", fn, p, ev=EV[fn]["final"], extra=dict(config=SEL[chosen]["name"], n_features=len(final_feats))))
    for sd in (1, 2, 3):
        ps, _ = M.fit_predict(chosen, SEL[chosen], final_feats, d["tr"], d["va"], seed=sd)
        srows.append(dict(fold=fn, seed=sd, pr_auc=E.average_precision_score(d["y"], ps)))
final_tbl = pd.DataFrame(frows); final_tbl.to_csv(OUT / "final_model_by_fold.csv", index=False)
pd.DataFrame(srows).to_csv(OUT / "final_seed_stability.csv", index=False)

# ---- 7. operational view of the final model: top-K (p-rank and EV-rank) + EV>0 threshold rule ----
orows, trows, fail = [], [], []
for fn, d in FD.items():
    if chosen == "RULE": break
    p = FINAL[fn]; ev = EV[fn]["final"]
    for rk, sc in [("rank_by_probability", p), ("rank_by_expected_value", ev)]:
        orows.append(dict(fold=fn, ranking=rk, **E.topk_stats(d["y"], d["amt"], sc, d["jit"], d["k"]),
                          oracle_stopped_per_check=E.oracle_per_check(d["y"], d["amt"], d["k"])))
    for rr in ("rule_prior",):
        orows.append(dict(fold=fn, ranking=rr, **E.topk_stats(d["y"], d["amt"], SC[fn][rr], d["jit"], d["k"]),
                          oracle_stopped_per_check=E.oracle_per_check(d["y"], d["amt"], d["k"])))
    for contact in (False, True):
        trows.append(dict(fold=fn, contact_cost_charged=contact, capacity_k=d["k"], **E.threshold_stats(d["y"], d["amt"], p, contact)))
pd.DataFrame(orows).to_csv(OUT / "topk_economics.csv", index=False)
pd.DataFrame(trows).to_csv(OUT / "ev_threshold_rule.csv", index=False)

# failure cases on headline fold
if chosen != "RULE":
    d = FD["B_test_like"]; va = d["va"].copy(); va["score"] = FINAL["B_test_like"]; va["amount"] = d["amt"]; va["y"] = d["y"]
    va["rank"] = np.empty(len(va), int); o = E.order(va.score.values, d["jit"]); va.loc[va.index[o], "rank"] = np.arange(1, len(va) + 1)
    cols = ["claim_id", "partner_id", "submitted_at", "amount", "score", "rank", "y", "customer_prior_claims", "auto_approved", "p_rate_eb"]
    fp = va[(va["rank"] <= d["k"]) & (va.y == 0)][cols].assign(kind="false_alarm_in_top_K")
    fn_ = va[(va["rank"] > d["k"]) & (va.y == 1)][cols].assign(kind="missed_fraud")
    pd.concat([fp, fn_]).sort_values(["kind", "rank"]).to_csv(OUT / "failure_cases_headline_B.csv", index=False)

# ---- 8. bootstrap (partner-cluster) on all folds; models: rules, 3 families (base), final ----
ci_all = []
for fn, d in FD.items():
    names = list(SC[fn].keys())
    boot = E.cluster_bootstrap(d["y"], d["amt"], d["part"], d["jit"], d["k"], SC[fn], EV[fn], reps=REPS, seed=42)
    ci = E.summarise_ci(boot, names, ["pr_auc", "roc_auc", "k_precision", "k_recall", "k_stopped_per_check", "k_net_per_check", "ev_stopped_per_check", "ev_net_per_check"])
    ci.insert(0, "fold", fn); ci_all.append(ci)
pd.concat(ci_all).to_csv(OUT / "bootstrap_ci.csv", index=False)

# ---- 8b. diagnostics: WHY is the headline fold hard? (no new folds; reuses A/B/C) ----
drows = []
for fn, d in FD.items():
    va = d["va"]; y = d["y"]; fr = y == 1
    known = va.p_lab_fraud.values > 0                       # partner had >=1 known fraud at the freeze date
    drows.append(dict(fold=fn, n_fraud=int(fr.sum()), frauds_at_partners_with_known_fraud=int((fr & known).sum()),
                      share_frauds_at_unseen_partners=round(float((fr & ~known).sum() / fr.sum()), 3),
                      fraud_rate_partners_with_known_fraud=round(float(y[known].mean()), 4) if known.sum() else np.nan,
                      fraud_rate_other_partners=round(float(y[~known].mean()), 4),
                      n_partners_with_fraud_in_val=int(va.loc[fr, "partner_id"].nunique()),
                      top5_partner_share_of_fraud=round(float(va.loc[fr, "partner_id"].value_counts().head(5).sum() / fr.sum()), 3)))
pd.DataFrame(drows).to_csv(OUT / "diagnostic_unseen_fraud_partners.csv", index=False)
hrows = []
if chosen != "RULE":
    for fn in ("B_test_like", "C_june"):
        d = FD[fn]; t = pd.to_datetime(d["va"].submitted_at); vs = pd.Timestamp(S.FOLDS[[f["name"] for f in S.FOLDS].index(fn)]["val_start"])
        half = np.where(t < vs + pd.Timedelta(days=15), "first_15d", "after_15d")
        for h in ("first_15d", "after_15d"):
            m = half == h
            if d["y"][m].sum() > 0:
                hrows.append(dict(fold=fn, window=h, n=int(m.sum()), n_fraud=int(d["y"][m].sum()), pr_auc=E.average_precision_score(d["y"][m], FINAL[fn][m]),
                                  prevalence=float(d["y"][m].mean())))
    d = FD["C_june"]; t = pd.to_datetime(d["va"].submitted_at); m1 = (t < pd.Timestamp("2026-06-16")).values
    wc = []
    for wname, w in [("uniform", None), ("post_may_x5", np.where(d["tr"].regime_post_may == 1, 5.0, 1.0)), ("post_may_x20", np.where(d["tr"].regime_post_may == 1, 20.0, 1.0))]:
        p, _ = M.fit_predict(chosen, SEL[chosen], final_feats, d["tr"], d["va"], weights=w, seed=SEED)
        for h, m in (("first_15d", m1), ("after_15d", ~m1)):
            wc.append(dict(family=chosen, weighting=wname, window=h, n_fraud=int(d["y"][m].sum()), pr_auc=E.average_precision_score(d["y"][m], p[m])))
    pd.DataFrame(wc).to_csv(OUT / "regime_weighting_foldC_halves.csv", index=False)
pd.DataFrame(hrows).to_csv(OUT / "diagnostic_horizon_halves.csv", index=False)

# ---- 9. record decisions ----
json.dump(dict(selected_hyperparams_on="A_pre_regime only", configs={k: v for k, v in SEL.items()},
               selection_rule="simplest family (lr<hgb<rf) whose mean PR-AUC on folds B,C is >=90% of the best family, and which beats the best rule baseline",
               mean_pr_auc_B_C={n: float(mean_bc(n)) for n in ["rule_prior", "rule_partner"] + FAMS},
               chosen_family=chosen, feature_decisions=decisions, final_features=final_feats,
               fold_K=FOLD_K, bootstrap=dict(kind="partner-cluster", reps=REPS, seed=42), seed=SEED, regime_weighting="see regime_weighting_foldC.csv"),
          open(OUT / "model_selection.json", "w"), indent=2, default=float)
print(f"finished in {time.time()-t0:.0f}s")
