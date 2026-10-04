"""Temporal check: does crash history go stale as streets change?

The label window is split in half: early = 2021-09-30 to 2024-03-30, late = 2024-03-31 to 2026-09-29.
Each predictor is scored on late-window crashes in the held-out ward (leave-one-ward-out, ranking only, so
train and validation wards are pooled for fitting):
  design, trained on early        - GBM on the 57 snapshot features, labels from early crashes only
  design, trained on late         - same, labels from late crashes in the training wards (same-time reference)
  design + history, trained late  - adds the early crash count as a feature. This is a crash-derived input, allowed
                                    here only because it strictly precedes the late label window; never in the ladder
  history only                    - early crash count (raw hot-spot ranking)
  LTS v1, length only             - references
It also reports how persistent sub-block crash locations are between the two halves.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.fetch_crashes import fetch_crashes  # noqa: E402
from src.evaluate import _auc, _ranking  # noqa: E402
from src.experiments import _clean_features  # noqa: E402
from src.features.build import feature_columns  # noqa: E402
from src.labels import level_from_count  # noqa: E402
from src.models.gbm import fit_gbm, gbm_proba  # noqa: E402
from src.split import ward_folds  # noqa: E402

CUT = pd.Timestamp("2024-03-31", tz="UTC")


def capture_and_auc(count: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    order = _ranking(score)
    n_top = max(1, (len(count) + 9) // 10)
    return _auc(count >= 1, score), float(count[order[:n_top]].sum() / count.sum())


def main():
    cfg = load_config()
    out = cfg["paths"]["output_dir"]
    t = pd.read_parquet(cfg["paths"]["table"]).reset_index(drop=True)
    key = t["dc_subblockkey"].astype(str).str.strip().str.lower()
    cr = fetch_crashes(cfg)
    cr = cr[cr["located"]]
    early = key.map(cr.loc[cr["REPORTDATE"] < CUT, "subblockkey"].value_counts()).fillna(0).astype(int).to_numpy()
    late = key.map(cr.loc[cr["REPORTDATE"] >= CUT, "subblockkey"].value_counts()).fillna(0).astype(int).to_numpy()

    hot = early >= 2
    persistence = {
        "crashes_early": int(early.sum()), "crashes_late": int(late.sum()),
        "p_late_given_early": float((late[early >= 1] >= 1).mean()),
        "p_late_given_no_early": float((late[early == 0] >= 1).mean()),
        "late_share_on_no_early_subblocks": float(late[early == 0].sum() / late.sum()),
        "hot_spots_early_ge2": int(hot.sum()),
        "hot_spots_with_no_late_crash": float((late[hot] == 0).mean()),
        "hot_cooled_no_bike_facility_now": float((t["ddot_bike_best"][hot & (late == 0)] == 0).mean()),
        "hot_stayed_no_bike_facility_now": float((t["ddot_bike_best"][hot & (late >= 1)] == 0).mean()),
        "spearman_early_late": float(pd.Series(early).corr(pd.Series(late), method="spearman")),
    }
    for k, v in persistence.items():
        print(f"{k:36s} {v:.3f}" if isinstance(v, float) else f"{k:36s} {v}")

    X = _clean_features(t[feature_columns(t)])
    Xh = X.assign(hist_early=early.astype(float))
    runs = [("design, trained on early", X, level_from_count(early)),
            ("design, trained on late", X, level_from_count(late)),
            ("design + history, trained on late", Xh, level_from_count(late))]
    scores = {name: np.full(len(t), np.nan) for name, _, _ in runs}
    folds = ward_folds(t["ward"].to_numpy(), n_val=cfg["split"]["n_val_wards"])
    with threadpool_limits(8):
        for f in folds:
            fit_idx = np.concatenate([f.train_idx, f.val_idx])
            for name, XX, y in runs:
                p = gbm_proba(fit_gbm(XX.iloc[fit_idx], y[fit_idx], cfg, seed=0), XX.iloc[f.test_idx])
                scores[name][f.test_idx] = p[:, 1] + 2 * p[:, 2]
            print(f"fold {f.test_ward} done", flush=True)
    scores["history only"] = early.astype(float)
    scores["LTS v1"] = t["ref_lts"].fillna(0).to_numpy(dtype=float)
    scores["length only"] = t["length_m"].fillna(0).to_numpy(dtype=float)

    targets = [("late", late, name) for name in scores] + [("early", early, "design, trained on early")]
    rows = []
    for window, count, name in targets:
        for f in folds:
            auc, cap = capture_and_auc(count[f.test_idx].astype(float), scores[name][f.test_idx])
            rows.append({"predictor": name, "scored_on": window, "test_ward": f.test_ward,
                         "auc_any": auc, "top10_capture_count": cap})
    folds_df = pd.DataFrame(rows)
    summary = folds_df.groupby(["scored_on", "predictor"], sort=False)[["auc_any", "top10_capture_count"]].agg(
        ["mean", "std"]).round(3)
    summary.columns = ["_".join(c) for c in summary.columns]
    print(summary.to_string())

    pd.Series(persistence).to_csv(out / "temporal_persistence.csv", header=["value"])
    summary.reset_index().to_csv(out / "temporal_check.csv", index=False)
    print(f"wrote {out / 'temporal_check.csv'} and {out / 'temporal_persistence.csv'}")


if __name__ == "__main__":
    main()
