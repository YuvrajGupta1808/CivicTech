"""Leave-one-ward-out CV runner: M0, M1, SPF (M2), GBM (M3), fusion (F) and any external model's OOF predictions."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.config import load_config
from src.decode import (apply_logit_bias, argmax, decode_thresholds, expected_level, fit_logit_bias,
                        fit_thresholds)
from src.evaluate import metrics as eval_metrics
from src.models.baselines import lts_from_raw, m0_proba, m1_lts
from src.models.fusion import fuse, select_weights
from src.models.gbm import fit_gbm, gbm_proba
from src.models.spf import fit_spf, spf_proba
from src.split import ward_folds

DECODERS = ("argmax", "bias", "thresholds")
FUSION_NAMES = ("fusion", "fusion_equal", "fusion_valsel")
MAX_CATEGORIES = 250  # HistGradientBoosting supports <= 255 categories per feature


def _norm_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Make X safe for HistGradientBoosting.

    bool -> float; object/string -> category; every categorical column -> integer-coded categories (pandas-3 `str`
    categories containing NA break sklearn's from_dtype). Codes are consistent across folds because X is sliced from
    one table. Categoricals with > 250 levels become plain float codes (HGB allows <= 255 categories).
    """
    X = X.copy()
    for c in X.columns:
        dt = X[c].dtype
        if dt == bool or str(dt) == "boolean":
            X[c] = X[c].astype(float)
            continue
        if dt == object or str(dt) in ("string", "str"):
            X[c] = X[c].astype("category")
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            codes = X[c].cat.codes
            if len(X[c].cat.categories) > MAX_CATEGORIES:
                print(f"[run_cv] WARNING: {c} has {len(X[c].cat.categories)} categories; using float codes")
                X[c] = codes.where(codes >= 0).astype(float)
            else:
                X[c] = pd.Categorical(codes.where(codes >= 0))
    return X


def _prep_external(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_k"] = _norm_key(df["dc_subblockkey"])
    return df.drop_duplicates(["_k", "test_ward", "split"], keep="last")


def _ext_probs(ext: pd.DataFrame, name: str, test_ward: int, split: str, keys: pd.Series) -> np.ndarray:
    sub = ext[(ext["test_ward"] == test_ward) & (ext["split"] == split)].set_index("_k")[["p0", "p1", "p2"]]
    P = sub.reindex(keys.to_numpy()).to_numpy(dtype=float)
    bad = int(np.isnan(P).any(1).sum())
    if bad:
        raise ValueError(f"external '{name}': {bad}/{len(keys)} {split} rows missing for test_ward={test_ward}")
    return P / P.sum(1, keepdims=True)


def _decode_all(Pv: np.ndarray, yv: np.ndarray, Pt: np.ndarray, step: float):
    """Fit the three decoders on val and apply to test. Returns ({decoder: yhat}, mu_test, (thr1, thr2))."""
    mu_v, mu_t = expected_level(Pv), expected_level(Pt)
    bias = fit_logit_bias(Pv, yv)
    thr = fit_thresholds(mu_v, yv, K=3, step=step)
    preds = {
        "argmax": argmax(Pt),
        "bias": apply_logit_bias(Pt, bias),
        "thresholds": decode_thresholds(mu_t, thr),
    }
    return preds, mu_t, (float(thr[0]), float(thr[1]))


def run_cv(table: pd.DataFrame, feature_cols: list, label_col: str = "level", cfg: dict | None = None,
           models=("m0", "m1", "spf", "gbm", "fusion", "fusion_equal", "fusion_valsel"), external: dict | None = None,
           count_col: str = "crash_count") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-ward-out CV. Returns (metrics_df, oof_df).

    models: any of m0, m1, spf, gbm, fusion, fusion_equal, fusion_valsel (external models are always reported).
    external: {name: DataFrame[dc_subblockkey, test_ward, split('val'|'test'), p0, p1, p2]}.
    Decoders for probabilistic models: argmax, bias (val-fitted logit bias), thresholds (val-fitted thresholds on mu).
    The score handed to the ranking metrics is the raw expected level mu for every decoder.
    """
    cfg = cfg or load_config()
    t_start = time.time()
    models = tuple(models)
    external = {k: _prep_external(v) for k, v in (external or {}).items()}
    table = table.reset_index(drop=True)
    step = cfg.get("decode", {}).get("threshold_grid_step", 0.01)

    keys = _norm_key(table["dc_subblockkey"])
    ward = table["ward"].to_numpy()
    y = table[label_col].to_numpy().astype(int)
    count = table[count_col].to_numpy(dtype=float)
    length = table["length_m"].to_numpy(dtype=float)
    X = _clean_features(table[list(feature_cols)])

    want_fusion = [m for m in FUSION_NAMES if m in models]
    need_spf = "spf" in models or bool(want_fusion)
    need_gbm = "gbm" in models or bool(want_fusion)
    lts_level = None
    if "m1" in models:
        # precomputed LTS (1..4) if the table carries it, else compute from the raw_<FIELD> passthrough columns
        lts_level = table["ref_lts"].astype(int) if "ref_lts" in table.columns else lts_from_raw(table)
    report = [m for m in models if m not in ("m0", "m1")] + list(external)  # probabilistic models reported
    fusion_weights = cfg["fusion"]["weights_with_xattn"] if "xattn" in external else cfg["fusion"]["weights"]

    folds = ward_folds(ward, cfg["split"]["n_val_wards"])
    metric_rows, oof_parts = [], []
    print(f"[run_cv] {len(table)} rows, {X.shape[1]} features, {len(folds)} folds, models={list(models)}"
          f" + external={list(external)}")

    for fi, fold in enumerate(folds):
        t0 = time.time()
        tr, va, te = fold.train_idx, fold.val_idx, fold.test_idx
        yv, yt = y[va], y[te]
        P: dict[str, tuple] = {}  # name -> (P_val, P_test)

        if need_spf:
            spf = fit_spf(table.iloc[tr], count[tr])
            P["spf"] = (spf_proba(spf, table.iloc[va]), spf_proba(spf, table.iloc[te]))
            spf_tag = f"{spf.family}{'' if spf.converged else '(not converged)'}"
        else:
            spf_tag = "-"
        if need_gbm:
            gb = fit_gbm(X.iloc[tr], y[tr], cfg, seed=0)
            P["gbm"] = (gbm_proba(gb, X.iloc[va]), gbm_proba(gb, X.iloc[te]))
        for name, ext in external.items():
            P[name] = (_ext_probs(ext, name, fold.test_ward, "val", keys.iloc[va]),
                       _ext_probs(ext, name, fold.test_ward, "test", keys.iloc[te]))

        member_names = [n for n in ("gbm", "spf", "xattn") if n in P] + [n for n in external if n != "xattn"]
        member_names = list(dict.fromkeys(n for n in member_names if n in P))
        weights_used: dict[str, dict] = {}
        if "fusion" in want_fusion:
            w = {n: v for n, v in fusion_weights.items() if n in P}
            weights_used["fusion"] = w
            P["fusion"] = (fuse({n: P[n][0] for n in w}, w), fuse({n: P[n][1] for n in w}, w))
        if "fusion_equal" in want_fusion:
            w = {n: 1.0 / len(member_names) for n in member_names}
            weights_used["fusion_equal"] = w
            P["fusion_equal"] = (fuse({n: P[n][0] for n in w}, w), fuse({n: P[n][1] for n in w}, w))
        if "fusion_valsel" in want_fusion:
            w = select_weights({n: P[n][0] for n in member_names}, yv, member_names)
            weights_used["fusion_valsel"] = w
            P["fusion_valsel"] = (fuse({n: P[n][0] for n in w}, w), fuse({n: P[n][1] for n in w}, w))

        base = dict(fold=fi, test_ward=fold.test_ward, n_test=len(te))

        def add_rows(name, preds, mu_t, Pt, thr=(np.nan, np.nan)):
            w = weights_used.get(name)
            for dec, yhat in preds.items():
                m = eval_metrics(yt, yhat, mu_t, count[te], length[te])
                metric_rows.append({**base, "model": name, "decoder": dec, **m,
                                    "weights": json.dumps(w) if w else None})
            na = pd.array([pd.NA] * len(te), dtype="Int64")
            oof_parts.append(pd.DataFrame({
                "dc_subblockkey": table["dc_subblockkey"].to_numpy()[te], "ward": ward[te], "model": name,
                "p0": Pt[:, 0], "p1": Pt[:, 1], "p2": Pt[:, 2], "mu": mu_t,
                "pred_argmax": pd.array(preds["argmax"], dtype="Int64"),
                "pred_bias": pd.array(preds["bias"], dtype="Int64") if "bias" in preds else na,
                "pred_thresholds": pd.array(preds["thresholds"], dtype="Int64") if "thresholds" in preds else na,
                "thr1": thr[0], "thr2": thr[1], "y": yt,
            }))

        if "m0" in models:
            Pt = m0_proba(len(te))
            add_rows("m0", {"argmax": argmax(Pt)}, expected_level(Pt), Pt)
        if "m1" in models:
            Pt, score = m1_lts(lts_level.iloc[te])
            add_rows("m1", {"argmax": argmax(Pt)}, score, Pt)
        for name in report:
            Pv, Pt = P[name]
            preds, mu_t, thr = _decode_all(Pv, yv, Pt, step)
            add_rows(name, preds, mu_t, Pt, thr)

        print(f"[run_cv] fold {fi + 1}/{len(folds)} test ward {fold.test_ward} val {fold.val_wards} "
              f"train {len(tr)} val {len(va)} test {len(te)} spf={spf_tag} ({time.time() - t0:.1f}s)")

    metrics_df = pd.DataFrame(metric_rows)
    oof_df = pd.concat(oof_parts, ignore_index=True)
    print(f"[run_cv] done in {time.time() - t_start:.1f}s")
    return metrics_df, oof_df
