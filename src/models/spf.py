"""M2: negative-binomial Safety Performance Function (Poisson GLM fallback).

crash_count ~ NB2(mu), log(mu) = X b + log(length_m).  P(0), P(1), P(>=2) come from the NB pmf with the fitted alpha.
Continuous predictors are centred on the training mean (slopes keep their raw-unit meaning, IRR = exp(coef) per unit).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import NegativeBinomial

MIN_LENGTH_M = 5.0
N_TOP_CLASSES = 4
SPEED_DEFAULT = 25.0

# continuous / ordinal predictors: (column, name used in the design)
_NUMERIC = ["ddot_log_aadt", "ddot_lanes_total", "ddot_speed_max", "ddot_bike_best"]


@dataclass
class SPFModel:
    family: str                      # "nb" or "poisson"
    converged: bool
    columns: list                    # design columns (excluding const), in order
    params: pd.Series                # coefficients incl. const
    bse: pd.Series
    pvalues: pd.Series
    alpha: float                     # NB dispersion (0.0 for poisson)
    fill: dict = field(default_factory=dict)      # train medians / means used for imputation / centring
    top_classes: list = field(default_factory=list)
    note: str = ""                   # why the fallback was used, if it was
    n_train: int = 0


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index)


def _cls(df: pd.DataFrame) -> pd.Series:
    if "ddot_fhwa_class" in df.columns:
        return df["ddot_fhwa_class"].astype("object").map(lambda v: None if pd.isna(v) else str(v))
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def _design(df: pd.DataFrame, fill: dict, top_classes: list) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    aadt = _num(df, "ddot_log_aadt")
    X["ddot_log_aadt"] = aadt.fillna(fill["ddot_log_aadt"]) - fill["ddot_log_aadt_mean"]
    X["aadt_missing"] = aadt.isna().astype(float)
    X["ddot_lanes_total"] = _num(df, "ddot_lanes_total").fillna(fill["ddot_lanes_total"]) - fill["ddot_lanes_total_mean"]
    X["ddot_speed_max"] = _num(df, "ddot_speed_max").fillna(SPEED_DEFAULT) - fill["ddot_speed_max_mean"]
    X["ddot_bike_best"] = _num(df, "ddot_bike_best").fillna(0.0) - fill["ddot_bike_best_mean"]
    c = _cls(df)
    for k in top_classes:
        X[f"fhwa_{k}"] = (c == k).astype(float)
    return X


def _offset(df: pd.DataFrame) -> np.ndarray:
    return np.log(np.clip(pd.to_numeric(df["length_m"], errors="coerce").fillna(MIN_LENGTH_M).to_numpy(float), MIN_LENGTH_M, None))


def fit_spf(df: pd.DataFrame, y_count) -> SPFModel:
    y = np.asarray(y_count, dtype=float)
    fill: dict = {}
    aadt = _num(df, "ddot_log_aadt")
    fill["ddot_log_aadt"] = float(aadt.median()) if aadt.notna().any() else 0.0
    fill["ddot_log_aadt_mean"] = float(aadt.fillna(fill["ddot_log_aadt"]).mean())
    lanes = _num(df, "ddot_lanes_total")
    fill["ddot_lanes_total"] = float(lanes.median()) if lanes.notna().any() else 2.0
    fill["ddot_lanes_total_mean"] = float(lanes.fillna(fill["ddot_lanes_total"]).mean())
    fill["ddot_speed_max_mean"] = float(_num(df, "ddot_speed_max").fillna(SPEED_DEFAULT).mean())
    fill["ddot_bike_best_mean"] = float(_num(df, "ddot_bike_best").fillna(0.0).mean())
    top = _cls(df).value_counts().head(N_TOP_CLASSES).index.tolist()  # remaining classes = reference ("other")

    X = _design(df, fill, top)
    # drop columns without variation in training (constant / all-missing predictors) to keep the fit identifiable
    keep = [c for c in X.columns if X[c].nunique() > 1]
    X = X[keep]
    Xc = sm.add_constant(X, has_constant="add")
    off = _offset(df)

    model, note = None, ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            nb = NegativeBinomial(y, Xc, offset=off, loglike_method="nb2")
            res = nb.fit(method="bfgs", maxiter=500, disp=0)
            conv = bool(res.mle_retvals.get("converged", False))
            alpha = float(res.params.iloc[-1])
            ok = conv and np.all(np.isfinite(res.params)) and np.all(np.isfinite(res.bse)) and np.isfinite(alpha) and alpha > 0
            if not ok:
                # second attempt from Poisson start values with Newton
                res2 = nb.fit(method="newton", maxiter=100, disp=0, start_params=np.r_[np.zeros(Xc.shape[1]), 0.5])
                conv2 = bool(res2.mle_retvals.get("converged", False))
                alpha2 = float(res2.params.iloc[-1])
                if conv2 and np.all(np.isfinite(res2.params)) and np.isfinite(alpha2) and alpha2 > 0:
                    res, conv, alpha, ok = res2, conv2, alpha2, True
            if ok:
                names = list(Xc.columns)
                model = SPFModel(
                    family="nb", converged=True, columns=keep,
                    params=pd.Series(np.asarray(res.params)[:-1], index=names),
                    bse=pd.Series(np.asarray(res.bse)[:-1], index=names),
                    pvalues=pd.Series(np.asarray(res.pvalues)[:-1], index=names),
                    alpha=alpha, fill=fill, top_classes=top, n_train=len(y),
                )
            else:
                note = f"NB did not converge (converged={conv}, alpha={alpha:.3g}); Poisson GLM fallback"
        except Exception as e:  # noqa: BLE001
            note = f"NB failed ({type(e).__name__}: {e}); Poisson GLM fallback"
        if model is None:
            res = sm.GLM(y, Xc, family=sm.families.Poisson(), offset=off).fit(maxiter=200)
            model = SPFModel(
                family="poisson", converged=bool(getattr(res, "converged", True)), columns=keep,
                params=res.params, bse=res.bse, pvalues=res.pvalues,
                alpha=0.0, fill=fill, top_classes=top, note=note, n_train=len(y),
            )
    return model


def spf_mu(model: SPFModel, df: pd.DataFrame) -> np.ndarray:
    """Predicted expected crash count per row (including the length offset)."""
    X = _design(df, model.fill, model.top_classes)[model.columns]
    eta = model.params["const"] + X.to_numpy(float) @ model.params[model.columns].to_numpy(float) + _offset(df)
    return np.exp(np.clip(eta, -20.0, 6.0))


def spf_proba(model: SPFModel, df: pd.DataFrame) -> np.ndarray:
    """(n, 3): P(0), P(1), P(>=2) under the fitted NB (or Poisson) distribution."""
    mu = spf_mu(model, df)
    if model.family == "nb" and model.alpha > 1e-8:
        n = 1.0 / model.alpha
        p = n / (n + mu)
        p0, p1 = stats.nbinom.pmf(0, n, p), stats.nbinom.pmf(1, n, p)
    else:
        p0, p1 = stats.poisson.pmf(0, mu), stats.poisson.pmf(1, mu)
    p2 = np.clip(1.0 - p0 - p1, 0.0, 1.0)
    P = np.column_stack([p0, p1, p2])
    return P / P.sum(1, keepdims=True)


def spf_coefficients(model: SPFModel) -> pd.DataFrame:
    """Coefficient table for the README: coef, IRR = exp(coef), 95% CI of IRR, std-err and p-value."""
    out = pd.DataFrame({"coef": model.params, "se": model.bse, "IRR": np.exp(model.params),
                        "IRR_lo": np.exp(model.params - 1.96 * model.bse),
                        "IRR_hi": np.exp(model.params + 1.96 * model.bse),
                        "p_value": model.pvalues})
    out.attrs.update(family=model.family, alpha=model.alpha, converged=model.converged, note=model.note,
                     n_train=model.n_train)
    return out
