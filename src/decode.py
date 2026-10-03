"""Ordinal decoding on class probabilities (port of AVB-Engage calibration.py).

Works on probability matrices P of shape (n, K). Differences from the original:
the objective is MACRO-F1 (ties broken by lower MAE), the threshold search is a
vectorised cumulative-count scan instead of a Python loop, and inputs are
probabilities rather than logits.
"""

import itertools

import numpy as np

_EPS = 1e-9
_TOL = 1e-12


# ---------------------------------------------------------------- validation
def _as_probs(P):
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] < 2:
        raise ValueError(f"P must have shape (n, K>=2); got {P.shape}")
    if not np.all(np.isfinite(P)):
        raise ValueError("P must contain only finite values")
    return P


def _as_labels(y, K):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D; got shape {y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(np.isfinite(y)) or not np.all(y == np.round(y)):
            raise ValueError("y must contain integer class indices")
    y = y.astype(np.int64)
    if y.size and (y.min() < 0 or y.max() > K - 1):
        raise ValueError(f"y must be between 0 and {K - 1}")
    return y


def _as_thresholds(thresholds, n_thresholds=None):
    thr = np.asarray(thresholds, dtype=np.float64)
    if thr.ndim != 1:
        raise ValueError(f"thresholds must be 1D; got shape {thr.shape}")
    if n_thresholds is not None and thr.shape[0] != n_thresholds:
        raise ValueError(f"thresholds must contain {n_thresholds} values; got {thr.shape[0]}")
    if not np.all(np.isfinite(thr)):
        raise ValueError("thresholds must contain only finite values")
    if np.any(np.diff(thr) <= 0):
        raise ValueError("thresholds must be strictly increasing")
    return thr


def _positive(value, name):
    if not np.isscalar(value) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number; got {value}")
    return float(value)


# ------------------------------------------------------------------- metrics
def confusion(y, yhat, K=3):
    """K x K confusion matrix, rows = truth, cols = prediction."""
    y = np.asarray(y, dtype=np.int64)
    yhat = np.asarray(yhat, dtype=np.int64)
    if y.shape != yhat.shape:
        raise ValueError("y and yhat must have the same shape")
    if y.size and (min(y.min(), yhat.min()) < 0 or max(y.max(), yhat.max()) > K - 1):
        raise ValueError(f"labels must be between 0 and {K - 1}")
    return np.bincount(y * K + yhat, minlength=K * K).reshape(K, K)


def f1_from_confusion(conf):
    """Per-class F1 from a confusion matrix; classes with no support or predictions get 0."""
    tp = np.diag(conf).astype(np.float64)
    denom = conf.sum(axis=0) + conf.sum(axis=1)
    return np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(y, yhat, K=3):
    """Unweighted mean F1 over classes 0..K-1 (zero for absent classes)."""
    return float(f1_from_confusion(confusion(y, yhat, K)).mean())


def _score(y, yhat, K):
    """(macro_f1, -mae) used to rank candidates; unchecked fast path."""
    conf = np.bincount(y * K + yhat, minlength=K * K).reshape(K, K)
    return float(f1_from_confusion(conf).mean()), -float(np.abs(y - yhat).mean())


def _select(f1, mae, coords):
    """Best candidate: max macro-F1, then min MAE, then the one nearest the tied centroid."""
    cand = np.flatnonzero(f1 >= f1.max() - _TOL)
    cand = cand[mae[cand] <= mae[cand].min() + _TOL]
    if cand.size > 1:
        pts = coords[cand]
        cand = cand[[int(((pts - pts.mean(axis=0)) ** 2).sum(axis=1).argmin())]]
    return int(cand[0])


# ------------------------------------------------------------------- decoding
def expected_level(P):
    """Expected ordinal level mu = P @ arange(K)."""
    P = _as_probs(P)
    return P @ np.arange(P.shape[1], dtype=np.float64)


def decode_thresholds(mu, thresholds):
    """yhat = number of thresholds <= mu (np.digitize semantics)."""
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 1 or not np.all(np.isfinite(mu)):
        raise ValueError("mu must be a finite 1D array")
    return np.digitize(mu, _as_thresholds(thresholds)).astype(np.int64)


def _best_on_grid(mu, y, K, grid, combos):
    """Evaluate every threshold combo (grid indices, shape (M, K-1)) via cumulative counts."""
    m = grid.size
    cell = np.searchsorted(grid, mu, side="right")  # 0..m; mu >= grid[a] iff cell >= a + 1
    hist = np.bincount(y * (m + 1) + cell, minlength=K * (m + 1)).reshape(K, m + 1)
    cum = np.zeros((K, m + 2))
    cum[:, 1:] = np.cumsum(hist, axis=1)
    bounds = np.empty((combos.shape[0], K + 1), dtype=np.int64)
    bounds[:, 0] = 0
    bounds[:, 1:K] = combos + 1
    bounds[:, K] = m + 1
    conf = cum[:, bounds[:, 1:]] - cum[:, bounds[:, :-1]]  # (true, combo, pred)
    tp = np.einsum("kmk->mk", conf)
    true = cum[:, -1]
    denom = conf.sum(axis=0) + true[None, :]
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0).mean(axis=1)
    dist = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :])
    mae = np.einsum("cmk,ck->m", conf, dist) / y.size
    return _select(f1, mae, grid[combos])


def fit_thresholds(mu, y, K=3, step=0.01, coarse_step=0.05, radius=0.1):
    """Increasing thresholds on mu maximising macro-F1 (ties: lower MAE).

    Coarse grid over [0, K-1] with `coarse_step`, then a local refine of +-`radius`
    around each coarse threshold with `step`. Returns an array of K-1 thresholds.
    """
    step = _positive(step, "step")
    coarse_step = _positive(coarse_step, "coarse_step")
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 1 or not np.all(np.isfinite(mu)):
        raise ValueError("mu must be a finite 1D array")
    y = _as_labels(y, K)
    if y.shape != mu.shape:
        raise ValueError("mu and y must have the same length")
    if y.size == 0:
        raise ValueError("mu must contain at least one value")

    # coarse: all strictly increasing combinations on the coarse grid
    coarse = np.round(np.arange(0.0, K - 1 + coarse_step / 2, coarse_step), 8)
    combos = np.array(list(itertools.combinations(range(coarse.size), K - 1)), dtype=np.int64)
    best = _best_on_grid(mu, y, K, coarse, combos)
    t0 = coarse[combos[best]]

    # fine: each threshold within +-radius of its coarse value, common grid of multiples of `step`
    ilo = int(np.floor(max(0.0, t0.min() - radius) / step + 1e-9))
    ihi = int(np.ceil(min(K - 1.0, t0.max() + radius) / step - 1e-9))
    fine = np.round(np.arange(ilo, ihi + 1) * step, 8)
    ranges = [np.flatnonzero(np.abs(fine - t) <= radius + 1e-9) for t in t0]
    grids = np.meshgrid(*ranges, indexing="ij")
    cand = np.stack([g.ravel() for g in grids], axis=1)
    cand = cand[np.all(np.diff(cand, axis=1) > 0, axis=1)]
    best = _best_on_grid(mu, y, K, fine, cand)
    return fine[cand[best]]


# ------------------------------------------------------------------ logit bias
def argmax(P):
    """Plain argmax decoding."""
    return _as_probs(P).argmax(axis=1).astype(np.int64)


def apply_logit_bias(P, bias):
    """Argmax of log(P + 1e-9) + bias; returns class indices."""
    P = _as_probs(P)
    bias = np.asarray(bias, dtype=np.float64)
    if bias.ndim != 1:
        raise ValueError(f"bias must be a 1D array; got shape {bias.shape}")
    if bias.shape[0] != P.shape[1]:
        raise ValueError(f"bias must contain one value per class; got {bias.shape[0]} and {P.shape[1]}")
    if not np.all(np.isfinite(bias)):
        raise ValueError("bias must contain only finite values")
    return (np.log(P + _EPS) + bias).argmax(axis=1).astype(np.int64)


def fit_logit_bias(P, y, search=(-3.0, 3.0), step=0.1, max_sweeps=5):
    """Per-class log-bias (class 0 fixed at 0) by coordinate search on macro-F1.

    Each sweep scans one class bias over `search` with the others fixed, moving only on a
    strict (macro-F1, -MAE) improvement; stops when a full sweep changes nothing.
    """
    P = _as_probs(P)
    K = P.shape[1]
    y = _as_labels(y, K)
    lo, hi = float(search[0]), float(search[1])
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
        raise ValueError(f"search must be a finite (min, max) with min <= max; got {search}")
    step = _positive(step, "step")
    values = np.round(np.arange(lo, hi + step / 2, step), 8)
    logp = np.log(P + _EPS)
    bias = np.zeros(K)
    cur = _score(y, logp.argmax(axis=1), K)
    for _ in range(max_sweeps):
        moved = False
        for k in range(1, K):
            f1 = np.empty(values.size)
            mae = np.empty(values.size)
            for i, v in enumerate(values):
                b = bias.copy()
                b[k] = v
                s = _score(y, (logp + b).argmax(axis=1), K)
                f1[i], mae[i] = s[0], -s[1]
            j = _select(f1, mae, values[:, None])
            new = (f1[j], -mae[j])
            if new[0] > cur[0] + _TOL or (new[0] >= cur[0] - _TOL and new[1] > cur[1] + _TOL):
                bias[k] = values[j]
                cur = new
                moved = True
        if not moved:
            break
    return bias
