"""Phase 1: clean -> dedupe -> label handling -> features -> folds -> artifacts. Run from project root:
    python scripts/run_phase1.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from kestrel import config as C, features as F, splits as S, pipeline as P
from kestrel.clean import load_all

C.ART.mkdir(exist_ok=True)
ds = load_all()
rep = ds["report"]

# ---- cleaned datasets ----
ds["model_set"].to_csv(C.ART / "model_set_clean.csv", index=False)
ds["undecided"].to_csv(C.ART / "train_undecided_blank_labels.csv", index=False)
ds["excluded"].to_csv(C.ART / "train_excluded_injected_text.csv", index=False)
ds["test"].to_csv(C.ART / "test_clean.csv", index=False)
ds["log"].to_csv(C.ART / "claims_log.csv", index=False)

# ---- features ----
Xtr = P.train_features(ds)
Xte = P.test_features(ds)
Xtr.to_csv(C.ART / "features_train.csv", index=False)
Xte.to_csv(C.ART / "features_test.csv", index=False)
rep["features_default"] = F.feature_names()
rep["features_optional_groups"] = {g: F.GROUPS[g] for g in ("partner_age", "serial", "desc")}

# ---- folds ----
rows, member = [], []
for fold in S.FOLDS:
    tr, va = P.fold_data(ds, fold, Xtr)
    k = S.top_k(len(va))
    rows.append(dict(fold=fold["name"], val_start=fold["val_start"], val_end=fold["val_end"],
                     label_cutoff=str(S.fold_cutoff(fold).date()), n_train=len(tr), train_fraud=int(tr.is_fraud.sum()),
                     n_val=len(va), val_fraud=int(va.is_fraud.sum()), val_fraud_rate=round(va.is_fraud.mean(), 4),
                     top_k=k, max_recall_at_k=round(min(k, va.is_fraud.sum()) / max(va.is_fraud.sum(), 1), 3),
                     val_auto_approved_share=round(va.auto_approved.mean(), 3)))
    member += [(c, fold["name"], "train") for c in tr.claim_id] + [(c, fold["name"], "val") for c in va.claim_id]
fd = pd.DataFrame(rows); fd.to_csv(C.ART / "fold_summary.csv", index=False)
pd.DataFrame(member, columns=["claim_id", "fold", "role"]).to_csv(C.ART / "fold_membership.csv", index=False)
rep["folds"] = rows

# ---- regime / covariate-shift snapshot ----
ms = ds["model_set"]; te = ds["test"]
rep["regime"] = {
    "train_post_may_labelled_rows": int(ms.regime_post_may.sum()),
    "train_post_may_fraud": int(ms[ms.regime_post_may == 1].is_fraud.sum()),
    "post_may_small_fraud_rate": round(float(ms[(ms.regime_post_may == 1) & (ms.claim_amount_inr < C.SMALL_CLAIM_INR)].is_fraud.mean()), 4),
    "pre_may_small_fraud_rate": round(float(ms[(ms.regime_post_may == 0) & (ms.claim_amount_inr < C.SMALL_CLAIM_INR)].is_fraud.mean()), 4),
    "test_small_share": round(float((te.claim_amount_inr < C.SMALL_CLAIM_INR).mean()), 3),
    "test_uninspected_share": round(float((te.partner_inspected == "N").mean()), 3),
    "train_uninspected_share_labelled": round(float((ms.partner_inspected == "N").mean()), 3),
    "post_may_small_but_inspected_Y_rows": int(((ms.regime_post_may == 1) & (ms.claim_amount_inr < C.SMALL_CLAIM_INR) & (ms.partner_inspected == "Y")).sum()),
}
rep["assumptions"] = dict(LAG_DAYS=C.LAG_DAYS, WINDOW_DAYS=C.WINDOW_DAYS, EB_STRENGTH=C.EB_STRENGTH,
                          test_label_cutoff=str(ms.submitted_at.max()))
(C.ART / "phase1_report.json").write_text(json.dumps(rep, indent=2, default=str))
print(json.dumps({k: v for k, v in rep.items() if k not in ("features_default", "features_optional_groups", "folds")}, indent=2, default=str))
print(fd.to_string(index=False))
print("train features", Xtr.shape, "test features", Xte.shape)
