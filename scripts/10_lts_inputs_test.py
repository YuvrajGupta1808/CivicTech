"""Fairness check: is the gain over LTS/BNA from learning on crashes, or from the extra facts?

Same GBM recipe and leave-one-ward-out folds, different inputs:
  lts_level_only   - RideScore LTS v1 level as the only input (LTS re-weighted by crash data)
  bna_level_only   - BNA segment stress as the only input
  lts_inputs       - the raw facts LTS uses: speed, lanes, bike facility, road class
  lts_inputs+net   - those plus junction / network geometry (net_*)
  all_features     - the full 57-feature model (M3)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.evaluate import summarise  # noqa: E402
from src.experiments import run_cv  # noqa: E402
from src.features.build import feature_columns  # noqa: E402
from src.models.bna import bna_stress  # noqa: E402

LTS_INPUTS = ["ddot_speed_max", "ddot_lanes_total", "ddot_bike_best", "ddot_dc_class"]
SHOW = ["macro_f1", "auc_any", "auc_repeat", "top10_capture_count"]


def main():
    cfg = load_config()
    out = cfg["paths"]["output_dir"]
    t = pd.read_parquet(cfg["paths"]["table"]).reset_index(drop=True)
    t["bna_stress"] = bna_stress(t).to_numpy().astype(float)
    t["ref_lts_f"] = t["ref_lts"].astype(float)
    net = [c for c in t.columns if c.startswith("net_")]
    variants = {
        "lts_level_only": ["ref_lts_f"],
        "bna_level_only": ["bna_stress"],
        "lts_inputs": LTS_INPUTS,
        "lts_inputs+net": LTS_INPUTS + net,
        "all_features": feature_columns(t),
    }
    rows = []
    for name, cols in variants.items():
        metrics, _ = run_cv(t, cols, label_col="level", cfg=cfg, models=("gbm",))
        m = metrics[metrics["decoder"] == "thresholds"].copy()
        m["model"] = name
        s = summarise(m)
        s.insert(1, "n_inputs", len(cols))
        rows.append(s)
        print(name, {k: round(float(s[f"{k}_mean"].iloc[0]), 3) for k in SHOW})
    res = pd.concat(rows, ignore_index=True)
    keep = ["model", "n_inputs"] + [f"{k}_{x}" for k in SHOW for x in ("mean", "std")]
    res[keep].to_csv(out / "lts_inputs_test.csv", index=False)
    print(res[keep].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
