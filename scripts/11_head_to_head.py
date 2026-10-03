"""Head-to-head: fusion (out-of-fold) vs RideScore LTS v1 vs BNA, per ward, per street type, at equal flag rate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.config import load_config  # noqa: E402
from src.models.bna import bna_stress  # noqa: E402

SCORES = {"model": "mu", "lts": "ref_lts", "bna": "bna"}


def auc(d, col, rng):
    if d["any"].nunique() < 2:
        return np.nan
    return roc_auc_score(d["any"], d[col] + rng.normal(0, 1e-9, len(d)))


def top10(d, col, rng):
    k = int(np.ceil(len(d) * 0.1))
    order = np.argsort(-(d[col].to_numpy() + rng.uniform(0, 1e-6, len(d))))[:k]
    return d["crash_count"].to_numpy()[order].sum() / d["crash_count"].sum()


def main():
    cfg = load_config()
    out = cfg["paths"]["output_dir"]
    rng = np.random.default_rng(0)
    t = pd.read_parquet(cfg["paths"]["table"]).reset_index(drop=True)
    t["bna"] = bna_stress(t).to_numpy()
    oof = pd.read_parquet(out / "oof_predictions.parquet")
    oof = oof[oof["model"] == "fusion"][["dc_subblockkey", "mu", "pred_thresholds"]]
    t = t.merge(oof, on="dc_subblockkey")
    t["any"] = (t["level"] >= 1).astype(int)
    fh = t["ddot_fhwa_class"].astype("string")
    groups = {f"ward {w}": t["ward"] == w for w in sorted(t["ward"].unique())}
    groups.update({
        "local streets (FHWA 7)": fh == "7", "collectors (FHWA 5-6)": fh.isin(["5", "6"]),
        "arterials (FHWA 3-4)": fh.isin(["3", "4"]),
        "has bike facility": t["ddot_bike_best"] > 0, "no bike facility": t["ddot_bike_best"] == 0,
    })
    rows = []
    for name, mask in groups.items():
        d = t[mask.fillna(False)]
        row = {"group": name, "n_subblocks": len(d), "crashes": int(d["crash_count"].sum())}
        for s, col in SCORES.items():
            row[f"auc_{s}"] = auc(d, col, rng)
            row[f"top10_{s}"] = top10(d, col, rng)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "head_to_head.csv", index=False)
    flag = []
    for name, m in [("LTS 4", t["ref_lts"] == 4), ("BNA stress 3", t["bna"] == 3)]:
        k = int(round(m.mean() * len(t)))
        top = t.nlargest(k, "mu")
        n2 = (t["level"] == 2).sum()
        flag.append({"rule": name, "share_flagged": m.mean(),
                     "rule_catches_2plus": (t[m]["level"] == 2).sum() / n2,
                     "model_same_share_catches_2plus": (top["level"] == 2).sum() / n2})
    pd.DataFrame(flag).to_csv(out / "equal_flag_rate.csv", index=False)
    with pd.option_context("display.width", 200):
        print(res.round(3).to_string(index=False))
        print(pd.DataFrame(flag).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
