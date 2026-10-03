"""F: probability-level fusion of M3 (GBM), M2 (SPF) and optionally M4 (cross-attention)."""
from __future__ import annotations

import itertools

import numpy as np

from src.decode import decode_thresholds, expected_level, fit_thresholds, macro_f1


def fuse(probs: dict, weights: dict) -> np.ndarray:
    """Weighted mean of the members' probability matrices; weights renormalised over members present."""
    names = [n for n in weights if n in probs and weights[n] > 0]
    if not names:
        raise ValueError(f"no overlap between probs {list(probs)} and weights {weights}")
    w = np.array([weights[n] for n in names], dtype=float)
    w = w / w.sum()
    P = sum(wi * np.asarray(probs[n], dtype=float) for wi, n in zip(w, names))
    return P / P.sum(1, keepdims=True)


def weight_grid(names, step: float = 0.1) -> list:
    """All weight vectors on the simplex with the given step, as dicts {name: weight}."""
    names = list(names)
    n = int(round(1.0 / step))
    grid = []
    for combo in itertools.product(range(n + 1), repeat=len(names) - 1):
        last = n - sum(combo)
        if last < 0:
            continue
        grid.append({nm: c / n for nm, c in zip(names, (*combo, last))})
    return grid


def macro_f1_thresholded(P_val: np.ndarray, y_val, step: float = 0.01) -> float:
    """Val macro-F1 after threshold decoding fitted on the same val rows."""
    mu = expected_level(P_val)
    thr = fit_thresholds(mu, y_val, K=3, step=step)
    return macro_f1(y_val, decode_thresholds(mu, thr))


def select_weights(probs_val: dict, y_val, names, objective=None, step: float = 0.1) -> dict:
    """Simplex grid search for the weights maximising `objective(P_val, y_val)`.

    Default objective = validation macro-F1 after threshold decoding.  Ties go to the weights closest to equal.
    """
    names = [n for n in names if n in probs_val]
    y_val = np.asarray(y_val).astype(int)
    if objective is None or objective == "macro_f1":
        objective = macro_f1_thresholded
    equal = 1.0 / len(names)
    best, best_key = None, None
    for w in weight_grid(names, step):
        score = objective(fuse(probs_val, w), y_val)
        key = (round(score, 10), -sum((v - equal) ** 2 for v in w.values()))
        if best_key is None or key > best_key:
            best, best_key = w, key
    return best
