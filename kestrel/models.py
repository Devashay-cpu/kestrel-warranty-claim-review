"""Model factories for Phase 2 (all consume the Phase 1 feature frames; categoricals are string columns)."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from . import features as F

COUNT_COLS = {"p_claims_7d", "p_claims_30d", "p_lab_n", "p_lab_fraud", "p_lab_n90", "p_lab_fraud90",
              "serial_prior_uses", "partner_age_days"}

# small pre-specified grids; the config is chosen on fold A ONLY (never on the headline window)
GRIDS = {
    "lr":  [dict(name="C=0.02", C=0.02), dict(name="C=0.1", C=0.1), dict(name="C=0.5", C=0.5)],
    "hgb": [dict(name="d2_i150", max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=40),
            dict(name="d3_i150", max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40),
            dict(name="d3_i300_slow", max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=80)],
    "rf":  [dict(name="leaf5", min_samples_leaf=5), dict(name="leaf20", min_samples_leaf=20), dict(name="leaf50", min_samples_leaf=50)],
}
SIMPLICITY_ORDER = ["lr", "hgb", "rf"]


def make_model(kind, feats, params, seed=0):
    cats = [f for f in feats if f in F.CAT_FEATURES]
    nums = [f for f in feats if f not in F.CAT_FEATURES]
    p = {k: v for k, v in params.items() if k != "name"}
    if kind == "lr":
        cnt = [f for f in nums if f in COUNT_COLS]; oth = [f for f in nums if f not in COUNT_COLS]
        pre = ColumnTransformer([
            ("cnt", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("sc", StandardScaler())]), cnt),
            ("num", StandardScaler(), oth),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
        clf = LogisticRegression(C=p["C"], max_iter=3000, random_state=seed)
    elif kind == "hgb":
        pre = ColumnTransformer([("num", "passthrough", nums),
                                 ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats)])
        catidx = list(range(len(nums), len(nums) + len(cats))) or None
        clf = HistGradientBoostingClassifier(l2_regularization=1.0, categorical_features=catidx,
                                             early_stopping=False, random_state=seed, **p)
    elif kind == "rf":
        pre = ColumnTransformer([("num", "passthrough", nums), ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
        clf = RandomForestClassifier(n_estimators=400, max_features="sqrt", n_jobs=-1, random_state=seed, **p)
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_predict(kind, params, feats, tr, va, weights=None, seed=0):
    m = make_model(kind, feats, params, seed)
    kw = {} if weights is None else {"clf__sample_weight": weights}
    m.fit(tr[feats], tr["is_fraud"], **kw)
    return m.predict_proba(va[feats])[:, 1], m
