"""Evaluation metrics for ordinal crash-risk predictions and fold summaries."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.decode import confusion

_EXCLUDE = ("fold", "test_ward", "seed")


def _auc(positive, score):
    """ROC AUC of `score` for a boolean target; NaN when only one class is present."""
    if positive.all() or not positive.any():
        return float("nan")
    return float(roc_auc_score(positive, score))


def _ranking(score, seed=0):
    """Row order by descending score with ties broken randomly (fixed seed)."""
    noise = np.random.default_rng(seed).random(len(score))
    return np.lexsort((noise, -np.asarray(score, dtype=np.float64)))


def metrics(y, yhat, score, crash_count=None, length_m=None, K=3):
    """Classification, ranking and capture metrics for one set of predictions."""
    y = np.asarray(y, dtype=np.int64)
    yhat = np.asarray(yhat, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n = y.size
    conf = confusion(y, yhat, K)
    tp = np.diag(conf).astype(np.float64)
    pred, true = conf.sum(axis=0), conf.sum(axis=1)
    f1_den = pred + true
    f1 = np.divide(2.0 * tp, f1_den, out=np.zeros(K), where=f1_den > 0)
    prec = np.divide(tp, pred, out=np.zeros(K), where=pred > 0)
    rec = np.divide(tp, true, out=np.zeros(K), where=true > 0)
    err = np.abs(y - yhat)

    out = {"accuracy": float(tp.sum() / n), "macro_f1": float(f1.mean())}
    for k in range(K):
        out[f"f1_{k}"] = float(f1[k])
    for k in range(K):
        out[f"prec_{k}"] = float(prec[k])
    for k in range(K):
        out[f"rec_{k}"] = float(rec[k])
    out["mae"] = float(err.mean())
    out["adjacent_acc"] = float((err <= 1).mean())
    out["auc_any"] = _auc(y >= 1, score)
    out["auc_repeat"] = _auc(y >= 2, score)

    if crash_count is not None:
        cc = np.asarray(crash_count, dtype=np.float64)
        total = cc.sum()
        order = _ranking(score)
        n_top = max(1, (n + 9) // 10)  # ceil(10% of rows), integer arithmetic
        out["top10_capture_count"] = float(cc[order[:n_top]].sum() / total) if total > 0 else float("nan")
        if length_m is not None:
            length = np.asarray(length_m, dtype=np.float64)[order]
            cum = np.cumsum(length)
            target = 0.1 * cum[-1]
            stop = int(np.searchsorted(cum, target - 1e-9 * cum[-1], side="left"))
            out["top10_capture_length"] = (
                float(cc[order[: stop + 1]].sum() / total) if total > 0 else float("nan")
            )
    out["n"] = int(n)
    out["n_pos2"] = int((y >= 2).sum())
    return out


def majority_reference(y, crash_count=None, length_m=None, K=3):
    """Metrics of always predicting class 0 with a constant score (the baseline to beat)."""
    y = np.asarray(y, dtype=np.int64)
    zeros = np.zeros(y.size, dtype=np.int64)
    return metrics(y, zeros, np.zeros(y.size), crash_count, length_m, K)


def summarise(rows, by=("model", "decoder")):
    """Mean and std across folds per metric: columns `<metric>_mean`, `<metric>_std`."""
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    by = list(by)
    missing = [c for c in by if c not in df.columns]
    if missing:
        raise KeyError(f"missing grouping columns: {missing}")
    cols = [
        c for c in df.columns
        if c not in by and c not in _EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]
    agg = df.groupby(by, sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()
