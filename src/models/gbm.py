"""M3: AVB-Engage GBM recipe -- 3 HistGradientBoostingClassifiers, class weights 1/sqrt(count), mean probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

K = 3


def _sample_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=K).astype(float)
    w_class = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1.0)), 0.0)
    w = w_class[y]
    return w / w.mean()


def fit_gbm(X: pd.DataFrame, y, cfg: dict, seed: int = 0) -> list:
    y = np.asarray(y).astype(int)
    sw = _sample_weight(y)
    g = cfg["gbm"]
    models = []
    for i, (lr, it) in enumerate(g["fits"]):
        m = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=it, max_leaf_nodes=g.get("max_leaf_nodes", 15),
            l2_regularization=g.get("l2_regularization", 1.0), early_stopping=False,
            random_state=seed + i, categorical_features="from_dtype",
        )
        models.append(m.fit(X, y, sample_weight=sw))
    return models


def _proba3(m, X) -> np.ndarray:
    p = m.predict_proba(X)
    out = np.zeros((len(p), K))
    out[:, np.asarray(m.classes_, dtype=int)] = p
    return out


def gbm_proba(models: list, X: pd.DataFrame) -> np.ndarray:
    return np.mean([_proba3(m, X) for m in models], axis=0)


def permutation_importance_groups(models: list, X: pd.DataFrame, y, groups: dict, n_repeats: int = 3,
                                  seed: int = 0) -> pd.Series:
    """Mean increase in log-loss when the columns of one group are permuted jointly (same row shuffle)."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    labels = list(range(K))

    def ll(Xm):
        return log_loss(y, np.clip(gbm_proba(models, Xm), 1e-9, 1.0), labels=labels)

    base = ll(X)
    out = {}
    for name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].iloc[perm].set_axis(X.index, axis=0)
            deltas.append(ll(Xp) - base)
        out[name] = float(np.mean(deltas))
    return pd.Series(out, name="logloss_increase").sort_values(ascending=False)
