"""Loading, normalising, de-duplicating. Row-level logic here is shared by training, test and the API."""
import re
import numpy as np
import pandas as pd
from . import config as C


def normalise_serial(s) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()


def serial_format(s) -> str:
    s = str(s)
    if s.startswith(" "):
        return "lead_space"
    if "-" in s:
        return "hyphen"
    return "plain"


def prepare_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic row-level normalisation (no cross-row info, no labels). Safe for a single API record."""
    d = df.copy()
    d["submitted_at"] = pd.to_datetime(d["submitted_at"]).astype("datetime64[ns]")  # IST, as displayed
    d["partner_id"] = d["partner_id"].astype(str).str.strip()
    d["serial_norm"] = d["product_serial"].map(normalise_serial)
    d["serial_format"] = d["product_serial"].map(serial_format)
    d["serial_lower"] = d["product_serial"].astype(str).str.contains("[a-z]").astype(int)
    desc = d["claim_description"].fillna("").astype(str).str.strip()
    d["desc_suspect"] = ~desc.isin(C.ALLOWED_DESCRIPTIONS)            # injected / unknown text
    d["desc_cat"] = np.where(d["desc_suspect"], "other", desc)        # free text is never used beyond this
    note = d["inspector_note"].fillna("").astype(str).str.strip()
    d["has_note"] = (note != "").astype(int)
    d["note_cat"] = np.where(note == "", "none", np.where(note.isin(C.ALLOWED_NOTES), note, "other"))
    return d


def dedupe(df: pd.DataFrame):
    """One row per claim_id: keep the FIRST submission; label = max of non-null labels (NaN if all NaN)."""
    d = df.sort_values(["submitted_at", "claim_id"], kind="stable").reset_index(drop=True)
    n_rows = d.groupby("claim_id").size()
    stats = {"rows_in": len(d), "claim_ids": int(d.claim_id.nunique()), "ids_with_resubmits": int((n_rows > 1).sum())}
    if "is_fraud" in d:
        lab = d.groupby("claim_id")["is_fraud"].agg(lambda s: s.max() if s.notna().any() else np.nan)
        nun = d.groupby("claim_id")["is_fraud"].nunique()
        stats["label_conflicts_between_resubmits"] = int((nun > 1).sum())
    keep = d.drop_duplicates("claim_id", keep="first").copy()
    if "is_fraud" in d:
        keep["is_fraud"] = keep["claim_id"].map(lab)
    keep["n_submissions"] = keep["claim_id"].map(n_rows)   # audit only; NOT a feature (absent in test)
    return keep.reset_index(drop=True), stats


def load_all(raw=C.RAW):
    """Returns dict of cleaned frames + a cleaning report (dict)."""
    tr_raw = pd.read_csv(raw / "train.csv")
    te_raw = pd.read_csv(raw / "test_unlabelled.csv")
    partners = pd.read_csv(raw / "partners.csv", parse_dates=["onboarded_date"])
    products = pd.read_csv(raw / "products.csv")
    rep = {"train_raw_rows": len(tr_raw), "test_raw_rows": len(te_raw)}

    tr_p = prepare_rows(tr_raw)
    injected_ids = sorted(tr_p.loc[tr_p.desc_suspect, "claim_id"].unique())
    rep["injected_text_claim_ids"] = injected_ids
    rep["injected_text_rows"] = int(tr_p.desc_suspect.sum())

    train, st = dedupe(tr_p)
    rep.update({f"dedupe_{k}": v for k, v in st.items()})
    train["label_status"] = np.where(train.is_fraud.isna(), "undecided", "labelled")
    train["excluded_injected"] = train.claim_id.isin(injected_ids)
    train["regime_post_may"] = (train.submitted_at >= C.REGIME_DATE).astype(int)

    test_p = prepare_rows(te_raw)
    test, tst = dedupe(test_p)
    rep["test_duplicate_ids"] = tst["ids_with_resubmits"]
    rep["test_suspect_description_rows"] = int(test.desc_suspect.sum())
    test["regime_post_may"] = (test.submitted_at >= C.REGIME_DATE).astype(int)

    model_set = train[(train.label_status == "labelled") & (~train.excluded_injected)].copy()
    model_set["is_fraud"] = model_set["is_fraud"].astype(int)
    undecided = train[train.label_status == "undecided"].copy()
    excluded = train[train.excluded_injected].copy()
    rep.update({
        "train_unique_claims": len(train), "model_set_rows": len(model_set),
        "model_set_fraud": int(model_set.is_fraud.sum()),
        "model_set_fraud_rate": round(float(model_set.is_fraud.mean()), 5),
        "undecided_claims": len(undecided), "undecided_sources": undecided.source.value_counts().to_dict(),
        "excluded_injected_claims": len(excluded), "excluded_injected_labels": {str(k): int(v) for k, v in excluded.is_fraud.value_counts(dropna=False).items()},
        "test_rows": len(test),
        "id_overlap_train_test": int(len(set(train.claim_id) & set(test.claim_id))),
    })

    # claim log used ONLY for label-free, strictly-earlier volume features (includes undecided + excluded claims)
    cols = ["claim_id", "partner_id", "submitted_at", "claim_amount_inr", "serial_norm"]
    log = pd.concat([train[cols], test[cols]], ignore_index=True).sort_values(["submitted_at", "claim_id"], kind="stable").reset_index(drop=True)
    log["is_small"] = (log.claim_amount_inr < C.SMALL_CLAIM_INR).astype(int)
    return dict(train=train, test=test, model_set=model_set, undecided=undecided, excluded=excluded,
                log=log, partners=partners, products=products, report=rep)
