"""Time-based validation. Training rows are only those whose label would be KNOWN at val_start (embargo = LAG_DAYS).
No random split anywhere. Resubmits are already collapsed to one row per claim_id, so no claim_id straddles a split."""
import pandas as pd
from . import config as C

FOLDS = [
    dict(name="A_pre_regime", val_start="2026-02-01", val_end="2026-05-01"),  # 3 months, old regime (sanity only)
    dict(name="B_test_like", val_start="2026-05-01", val_end="2026-07-01"),   # 2 months post-May, history frozen at start
    dict(name="C_june", val_start="2026-06-01", val_end="2026-07-01"),        # latest month, includes some post-May training
]


def fold_cutoff(fold):
    """Last submission date whose label is treated as known when the validation window opens."""
    return pd.Timestamp(fold["val_start"]) - pd.Timedelta(days=C.LAG_DAYS)


def fold_masks(df, fold):
    t = df["submitted_at"]
    train = t <= fold_cutoff(fold)
    val = (t >= pd.Timestamp(fold["val_start"])) & (t < pd.Timestamp(fold["val_end"]))
    return train, val


def top_k(n_rows):
    """Investigation capacity expressed for a window of n_rows claims: 40 per ~750."""
    return max(1, round(n_rows * C.CAPACITY_PER_MONTH / C.CLAIMS_PER_MONTH))
