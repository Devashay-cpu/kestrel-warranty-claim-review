"""Glue used by the Phase 1 script, Phase 2 modelling and (later) the API."""
import pandas as pd
from . import config as C, features as F, splits as S
from .clean import load_all

EV_COLS = ["partner_id", "submitted_at", "is_fraud"]


def events_of(model_set):
    return model_set[EV_COLS].sort_values("submitted_at").reset_index(drop=True)


def train_features(ds):
    """Rolling point-in-time features for every labelled training claim."""
    ms = ds["model_set"]
    return F.build_features(ms, ds["partners"], ds["products"], ds["log"], events_of(ms), ("rolling",))


def test_features(ds):
    """Test claims see partner label history frozen at the end of the labelled data."""
    ms = ds["model_set"]
    return F.build_features(ds["test"], ds["partners"], ds["products"], ds["log"], events_of(ms),
                            ("frozen", ms.submitted_at.max()))


def fold_data(ds, fold, X_roll=None):
    """(train_df, val_df) of feature frames for one time-based fold. Val history is frozen at the fold cutoff."""
    ms = ds["model_set"]
    X_roll = train_features(ds) if X_roll is None else X_roll
    tr_m, va_m = S.fold_masks(ms, fold)
    cutoff = S.fold_cutoff(fold)
    ev = events_of(ms[ms.submitted_at <= cutoff])
    val = F.build_features(ms[va_m], ds["partners"], ds["products"], ds["log"], ev, ("frozen", cutoff))
    return X_roll[tr_m.values].reset_index(drop=True), val
