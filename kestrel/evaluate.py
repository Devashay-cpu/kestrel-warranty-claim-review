"""Metrics: ranking (PR-AUC), capacity (top-K), rupee economics (ops-policy s4), accuracy baseline, cluster bootstrap."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from . import config as C


def order(score, jit):
    """Rank descending by score; ties broken by a fixed random jitter (never by label/amount)."""
    return np.lexsort((np.asarray(jit), -np.asarray(score, dtype=float)))


def expected_value(p, amt):
    """Expected rupees of checking a claim: stop the full amount if fraud, pay Rs 380 goodwill if genuine."""
    return p * amt - (1 - p) * C.GOODWILL_INR


def topk_stats(y, amt, rank_score, jit, k):
    o = order(rank_score, jit)[:k]
    tp = int(y[o].sum()); fp = k - tp; pos = int(y.sum())
    stopped = float((amt[o] * y[o]).sum())
    net = stopped - C.GOODWILL_INR * fp
    return dict(k=k, tp=tp, fp=fp, precision=tp / k, recall=tp / max(pos, 1), stopped_inr=stopped,
                stopped_per_check=stopped / k, net_per_check=net / k,
                net_per_check_after_contact=net / k - C.CONTACT_INR,
                accuracy=1 - (fp + pos - tp) / len(y))


def oracle_per_check(y, amt, k):
    a = np.sort(amt[y == 1])[::-1][:k]
    return float(a.sum() / k)


def core_metrics(y, amt, score, jit, k, ev=None):
    """Scalar metrics used by the tables AND the bootstrap. ev = expected-value ranking score (optional)."""
    prev = y.mean()
    ap = average_precision_score(y, score)
    out = dict(pr_auc=ap, roc_auc=roc_auc_score(y, score), prevalence=prev, pr_lift=ap / prev)
    t = topk_stats(y, amt, score, jit, k)
    out.update({f"k_{a}": b for a, b in t.items() if a != "k"})
    if ev is not None:
        e = topk_stats(y, amt, ev, jit, k)
        out.update({f"ev_{a}": b for a, b in e.items() if a not in ("k", "accuracy")})
    return out


def threshold_stats(y, amt, p, contact=False):
    """Operational decision rule: check a claim iff expected value > 0 (> Rs 260 if contact cost is charged).
    Requires calibrated p; reported separately from ranking quality."""
    ev = expected_value(p, amt)
    flag = ev > (C.CONTACT_INR if contact else 0)
    n = int(flag.sum()); tp = int(y[flag].sum()); pos = int(y.sum())
    stopped = float((amt[flag] * y[flag]).sum()); fp = n - tp
    return dict(n_flagged=n, precision=tp / n if n else np.nan, recall=tp / max(pos, 1), stopped_inr=stopped,
                stopped_per_check=stopped / n if n else np.nan,
                net_inr=stopped - C.GOODWILL_INR * fp - (C.CONTACT_INR * n if contact else 0))


def cluster_bootstrap(y, amt, partner, jit, k, scores, evs=None, reps=1000, seed=42):
    """Partner-cluster bootstrap (fraud clusters at outlets, so rows are not independent).
    scores: {name: score}; evs: {name: ev score}. Returns per-rep metric dict and paired diffs vs 'rule_prior'."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(partner)
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    evs = evs or {}
    rows = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        yy = y[idx]
        if yy.sum() == 0:
            continue
        r = {}
        for name, s in scores.items():
            m = core_metrics(yy, amt[idx], s[idx], jit[idx], k, evs[name][idx] if name in evs else None)
            for a, b in m.items():
                r[(name, a)] = b
        rows.append(r)
    return pd.DataFrame(rows)


def summarise_ci(boot, names, metrics, baseline="rule_prior"):
    out = []
    for n in names:
        for m in metrics:
            if (n, m) not in boot: continue
            v = boot[(n, m)].dropna()
            row = dict(model=n, metric=m, mean=v.mean(), lo=v.quantile(0.025), hi=v.quantile(0.975))
            if baseline in names and n != baseline and (baseline, m) in boot:
                d = (boot[(n, m)] - boot[(baseline, m)]).dropna()
                row.update(diff_vs_baseline=d.mean(), diff_lo=d.quantile(0.025), diff_hi=d.quantile(0.975),
                           p_better=float((d > 0).mean()))
            out.append(row)
    return pd.DataFrame(out)
