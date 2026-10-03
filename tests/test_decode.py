import time

import numpy as np
import pandas as pd
import pytest

from src.decode import (
    apply_logit_bias,
    argmax,
    decode_thresholds,
    expected_level,
    fit_logit_bias,
    fit_thresholds,
    macro_f1,
)
from src.evaluate import majority_reference, metrics, summarise


def _imbalanced_problem(n=8000, seed=0):
    """91/6/3 labels; informative probabilities whose argmax is always class 0."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.91, 0.06, 0.03])
    s = y + rng.normal(0.0, 0.9, size=n)
    p1 = 0.30 / (1.0 + np.exp(-2.0 * (s - 0.8)))
    p2 = 0.20 / (1.0 + np.exp(-2.0 * (s - 1.8)))
    P = np.stack([1.0 - p1 - p2, p1, p2], axis=1)
    return P, y


def test_expected_level_and_decode_boundaries():
    P = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(expected_level(P), [0.0, 0.5, 2.0])
    mu = np.array([0.0, 0.49, 0.5, 1.49, 1.5, 2.0])
    np.testing.assert_array_equal(decode_thresholds(mu, [0.5, 1.5]), [0, 0, 1, 1, 2, 2])


def test_decode_thresholds_rejects_bad_thresholds():
    mu = np.array([0.1, 0.9])
    for thr, msg in [([1.0, 0.5], "strictly increasing"), ([0.5, np.nan], "finite")]:
        with pytest.raises(ValueError, match=msg):
            decode_thresholds(mu, thr)


def test_fit_thresholds_monotone_and_in_range():
    P, y = _imbalanced_problem()
    tau = fit_thresholds(expected_level(P), y)
    assert tau.shape == (2,)
    assert 0.0 <= tau[0] < tau[1] <= 2.0


def test_fit_thresholds_beats_argmax_macro_f1():
    P, y = _imbalanced_problem()
    assert np.all(argmax(P) == 0)  # premise: argmax never predicts a crash class
    mu = expected_level(P)
    tau = fit_thresholds(mu, y)
    base = macro_f1(y, argmax(P))
    fitted = macro_f1(y, decode_thresholds(mu, tau))
    assert fitted > base + 0.15


def test_fit_thresholds_matches_brute_force_on_coarse_grid():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=300)
    mu = np.clip(y + rng.normal(0, 0.7, size=300), 0, 2)
    tau = fit_thresholds(mu, y, step=0.05)
    grid = np.round(np.arange(0.0, 2.0001, 0.05), 8)
    best = max(
        macro_f1(y, decode_thresholds(mu, [a, b]))
        for i, a in enumerate(grid)
        for b in grid[i + 1:]
    )
    assert macro_f1(y, decode_thresholds(mu, tau)) >= best - 1e-12


def test_fit_thresholds_is_fast_on_8k_rows():
    P, y = _imbalanced_problem(n=8000)
    mu = expected_level(P)
    t0 = time.perf_counter()
    fit_thresholds(mu, y)
    assert time.perf_counter() - t0 < 2.0


def test_fit_thresholds_generic_k():
    rng = np.random.default_rng(2)
    y = rng.choice(4, size=400, p=[0.7, 0.15, 0.1, 0.05])
    mu = np.clip(y + rng.normal(0, 0.5, size=400), 0, 3)
    tau = fit_thresholds(mu, y, K=4)
    assert tau.shape == (3,) and np.all(np.diff(tau) > 0)
    assert macro_f1(y, decode_thresholds(mu, tau), K=4) > 0.5


def test_logit_bias_shifts_predictions():
    P, y = _imbalanced_problem()
    assert np.all(argmax(P) == 0)
    bias = fit_logit_bias(P, y)
    assert bias[0] == 0.0 and bias[1] > 0.0 and bias[2] > 0.0
    yhat = apply_logit_bias(P, bias)
    assert (yhat > 0).any()
    assert macro_f1(y, yhat) > macro_f1(y, argmax(P)) + 0.15


def test_apply_logit_bias_manual_shift_and_validation():
    P = np.array([[0.6, 0.4, 0.0], [0.7, 0.2, 0.1]])
    np.testing.assert_array_equal(apply_logit_bias(P, [0.0, 0.0, 0.0]), [0, 0])
    np.testing.assert_array_equal(apply_logit_bias(P, [0.0, 0.5, 0.0]), [1, 0])
    with pytest.raises(ValueError, match="one value per class"):
        apply_logit_bias(P, [0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        apply_logit_bias(P, [0.0, np.nan, 0.0])


def test_macro_f1_includes_absent_predicted_class_as_zero():
    y = np.array([0, 0, 1, 2])
    assert macro_f1(y, np.zeros(4, dtype=int)) == pytest.approx((2 * 2 / 6) / 3)


def _tiny():
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
    yhat = np.array([0, 0, 0, 0, 1, 2, 1, 0, 2, 1])
    score = np.array([0.1, 0.2, 0.15, 0.05, 0.6, 0.7, 0.5, 0.3, 0.9, 0.4])
    crash = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 3])
    length = np.array([10, 10, 10, 10, 2, 2, 2, 10, 2, 10], dtype=float)
    return y, yhat, score, crash, length


def test_metrics_hand_computed_tiny_case():
    y, yhat, score, crash, length = _tiny()
    m = metrics(y, yhat, score, crash, length)
    # confusion: TP = (4, 1, 1), predicted = (5, 3, 2), true = (6, 2, 2)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["f1_0"] == pytest.approx(8 / 11)
    assert m["f1_1"] == pytest.approx(0.4)
    assert m["f1_2"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx(179 / 330)
    assert (m["prec_0"], m["rec_0"]) == (pytest.approx(4 / 5), pytest.approx(4 / 6))
    assert (m["prec_1"], m["rec_1"]) == (pytest.approx(1 / 3), pytest.approx(1 / 2))
    assert (m["prec_2"], m["rec_2"]) == (pytest.approx(1 / 2), pytest.approx(1 / 2))
    assert m["mae"] == pytest.approx(0.5)  # errors 1, 2, 1, 1
    assert m["adjacent_acc"] == pytest.approx(0.9)
    assert m["auc_any"] == pytest.approx(18 / 24)
    assert m["auc_repeat"] == pytest.approx(13 / 16)
    assert m["top10_capture_count"] == pytest.approx(2 / 7)  # 1 of 10 rows: row 8
    # total length 68 -> 6.8 m: rows 8, 5, 4 reach 6 m, row 6 crosses it -> crashes 2 + 1
    assert m["top10_capture_length"] == pytest.approx(3 / 7)
    assert m["n"] == 10 and m["n_pos2"] == 2


def test_metrics_skips_capture_without_crash_count_and_handles_single_class():
    y = np.zeros(5, dtype=int)
    m = metrics(y, y, np.arange(5.0))
    assert "top10_capture_count" not in m and "top10_capture_length" not in m
    assert np.isnan(m["auc_any"]) and np.isnan(m["auc_repeat"])
    assert m["accuracy"] == 1.0


def test_majority_reference():
    y, _, _, crash, length = _tiny()
    m = majority_reference(y, crash, length)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["macro_f1"] == pytest.approx(0.75 / 3)
    assert m["auc_any"] == pytest.approx(0.5) and m["mae"] == pytest.approx(0.6)


def test_summarise_mean_std_columns():
    rows = [
        {"model": "gbm", "decoder": "thr", "fold": 0, "macro_f1": 0.4, "mae": 0.2},
        {"model": "gbm", "decoder": "thr", "fold": 1, "macro_f1": 0.6, "mae": 0.4},
        {"model": "spf", "decoder": "thr", "fold": 0, "macro_f1": 0.3, "mae": 0.1},
        {"model": "spf", "decoder": "thr", "fold": 1, "macro_f1": 0.3, "mae": 0.1},
    ]
    out = summarise(rows)
    assert {"model", "decoder", "macro_f1_mean", "macro_f1_std", "mae_mean", "mae_std"} <= set(out.columns)
    assert "fold_mean" not in out.columns
    gbm = out[out.model == "gbm"].iloc[0]
    assert gbm["macro_f1_mean"] == pytest.approx(0.5)
    assert gbm["macro_f1_std"] == pytest.approx(np.std([0.4, 0.6], ddof=1))
    assert isinstance(summarise(pd.DataFrame(rows)), pd.DataFrame)
