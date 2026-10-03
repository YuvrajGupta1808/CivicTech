"""BNA (bikescore-bna segment stress) vs RideScore LTS v1 vs our fusion model.

Usage: .venv/bin/python scripts/07_bna.py
Writes output/bna_vs_lts.csv, bna_vs_crashes.csv, bna_metrics.csv, bna_compare.png.

BNA as a model (nothing to fit; test-ward rows of the leave-one-ward-out folds only):
  score = 0 for stress 1, 1 for stress 3;  yhat = 0 (level 0) for stress 1, 1 (level 1) for stress 3.
  LTS-coarse uses the same mapping with LTS<=2 -> 0 and LTS>=3 -> 1. Neither binary predicts level 2, so
  accuracy / macro-F1 / rec_2 are not comparable with the 3-class models; AUCs and top-10% capture are.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import metrics, summarise  # noqa: E402
from src.models.bna import BNA_VERSION, bna_stress  # noqa: E402
from src.split import ward_folds  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output"
TABLE = Path.home() / "ridescore-data" / "subblock_table.parquet"
MUTED = ("#6b8fb3", "#c98a6b")  # comfortable / uncomfortable


def agreement(t, variant):
    """Long table: 2x2 coarse crosstab, 4x2 ref_lts x bna, and overall agreement, with sub-block and km shares."""
    rows = []
    n_all, km_all = len(t), t["length_m"].sum() / 1000

    def add(table, a, b, mask):
        rows.append({"variant": variant, "table": table, "ref_lts_group": a, "bna_stress": b,
                     "n_subblocks": int(mask.sum()), "length_km": t.loc[mask, "length_m"].sum() / 1000,
                     "share_of_subblocks": mask.sum() / n_all, "share_of_km": t.loc[mask, "length_m"].sum() / 1000 / km_all})

    lts_low = t["ref_lts"] <= 2
    for name, m in (("LTS<=2", lts_low), ("LTS>=3", ~lts_low)):
        for b in (1, 3):
            add("coarse_2x2", name, b, m & (t[f"bna_{variant}"] == b))
    for k in (1, 2, 3, 4):
        for b in (1, 3):
            add("ref_lts_x_bna", k, b, (t["ref_lts"] == k) & (t[f"bna_{variant}"] == b))
    agree = lts_low == (t[f"bna_{variant}"] == 1)
    add("agreement", "all", "agree", agree)
    add("agreement", "all", "disagree", ~agree)
    add("agreement", "LTS<=2 but BNA 3", 3, lts_low & (t[f"bna_{variant}"] == 3))
    add("agreement", "LTS>=3 but BNA 1", 1, ~lts_low & (t[f"bna_{variant}"] == 1))
    return pd.DataFrame(rows)


def crash_table(t, label, col, groups):
    """Crash statistics for each (name, mask) group of one classifier."""
    rows = []
    for name, mask in groups:
        d = t[mask]
        km = d["length_m"].sum() / 1000
        rows.append({"classifier": label, "group": name, "n_subblocks": len(d), "length_km": km,
                     "crashes": int(d["crash_count"].sum()), "crashes_per_km": d["crash_count"].sum() / km,
                     "share_any_crash": (d["crash_count"] > 0).mean(), "share_level_2": (d["level"] == 2).mean()})
    return rows


def fold_rows(t, folds, yhat, score, model, decoder):
    rows = []
    for f in folds:
        d = t.iloc[f.test_idx]
        m = metrics(d["level"], yhat[f.test_idx], score[f.test_idx], d["crash_count"], d["length_m"])
        rows.append({"model": model, "decoder": decoder, "test_ward": f.test_ward, **m})
    return rows


def main():
    t = pd.read_parquet(TABLE).reset_index(drop=True)
    for variant, src in (("fused", "fused"), ("osm", "osm")):
        t[f"bna_{variant}"] = bna_stress(t, src).to_numpy()
    t["bna"] = t["bna_fused"]
    print(f"BNA source: {BNA_VERSION}, real package rule table, segment stress only")
    print(f"n sub-blocks {len(t)}, km {t['length_m'].sum() / 1000:.0f}")
    for v in ("fused", "osm"):
        vc = t[f"bna_{v}"].value_counts().sort_index()
        km = t.groupby(f"bna_{v}")["length_m"].sum() / 1000
        print(f"stress distribution [{v}]: " + ", ".join(f"{k}: {n} ({n / len(t):.1%}, {km[k]:.0f} km)" for k, n in vc.items()))

    # (b) agreement with LTS
    agr = pd.concat([agreement(t, "fused"), agreement(t, "osm")], ignore_index=True)
    agr.to_csv(OUT / "bna_vs_lts.csv", index=False)
    for v in ("fused", "osm"):
        a = agr[(agr.variant == v) & (agr.table == "agreement") & (agr.ref_lts_group == "all") & (agr.bna_stress == "agree")].iloc[0]
        print(f"LTS(<=2) vs BNA(1) agreement [{v}]: {a.share_of_subblocks:.1%} of sub-blocks, {a.share_of_km:.1%} of km")
    print(pd.crosstab(t["ref_lts"], t["bna"]).to_string())

    # (c) crashes by stress, next to LTS-coarse and fusion
    oof = pd.read_parquet(OUT / "oof_predictions.parquet")
    fus = oof[oof["model"] == "fusion"].set_index("dc_subblockkey")["pred_thresholds"]
    t["fusion_level"] = t["dc_subblockkey"].map(fus)
    assert t["fusion_level"].notna().all()
    crows = []
    crows += crash_table(t, "BNA stress", "bna", [("stress 1 (comfortable)", t.bna == 1), ("stress 3 (uncomfortable)", t.bna == 3)])
    crows += crash_table(t, "BNA stress (OSM-only inputs)", "bna_osm",
                         [("stress 1 (comfortable)", t.bna_osm == 1), ("stress 3 (uncomfortable)", t.bna_osm == 3)])
    crows += crash_table(t, "LTS v1 coarse", "ref_lts", [("LTS <= 2", t.ref_lts <= 2), ("LTS >= 3", t.ref_lts >= 3)])
    crows += crash_table(t, "Fusion (OOF, thresholds)", "fusion_level",
                         [("predicted level 0", t.fusion_level == 0), ("predicted level >= 1", t.fusion_level >= 1)])
    ctab = pd.DataFrame(crows)
    ctab.to_csv(OUT / "bna_vs_crashes.csv", index=False)
    print(ctab.round(3).to_string(index=False))

    # (d) leave-one-ward-out metrics (test-ward rows only; nothing to fit)
    folds = ward_folds(t["ward"].to_numpy())
    rows = []
    bna_bin = (t["bna"] == 3).to_numpy().astype(int)
    bna_osm_bin = (t["bna_osm"] == 3).to_numpy().astype(int)
    lts_bin = (t["ref_lts"] >= 3).to_numpy().astype(int)
    rows += fold_rows(t, folds, bna_bin, bna_bin.astype(float), "bna", "stress3->level1")
    rows += fold_rows(t, folds, bna_osm_bin, bna_osm_bin.astype(float), "bna_osm_only", "stress3->level1")
    rows += fold_rows(t, folds, lts_bin, lts_bin.astype(float), "lts_coarse", "lts>=3->level1")
    rows = pd.DataFrame(rows)
    mf = pd.read_csv(OUT / "metrics_folds.csv")
    mf = mf[(mf["model"] == "fusion") & (mf["decoder"] == "thresholds")]
    keep = [c for c in rows.columns if c in mf.columns]
    rows = pd.concat([rows, mf[keep]], ignore_index=True)
    summ = summarise(rows)
    summ.to_csv(OUT / "bna_metrics.csv", index=False)
    rows.to_csv(OUT / "bna_metrics_folds.csv", index=False)
    cols = ["auc_any", "auc_repeat", "top10_capture_count", "top10_capture_length"]
    print(f"\nmean +- sd over {len(folds)} test wards")
    for _, r in summ.iterrows():
        print(f"{r['model']:14s} {r['decoder']:16s} " + "  ".join(f"{c}={r[c + '_mean']:.3f}+-{r[c + '_std']:.3f}" for c in cols))

    # (e) figure
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    groups = [("BNA stress", "BNA stress"), ("LTS v1 coarse", "LTS v1\n(<=2 vs >=3)"), ("Fusion (OOF, thresholds)", "Our model\n(level 0 vs >=1)")]
    x = np.arange(len(groups))
    for j, (lab, colour) in enumerate(zip(("comfortable / low risk", "uncomfortable / high risk"), MUTED)):
        vals = [ctab[ctab.classifier == g].iloc[j]["crashes_per_km"] for g, _ in groups]
        bars = ax.bar(x + (j - 0.5) * 0.36, vals, 0.34, label=lab, color=colour)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [g[1] for g in groups])
    ax.set_ylabel("crashes per km")
    ax.set_title("Crash rate by predicted comfort / risk class")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "bna_compare.png", dpi=150)
    plt.close(fig)

    # (f) summary
    s = summ.set_index("model")
    print("\nSUMMARY")
    print(f"BNA = {BNA_VERSION}; stress 1/3 = {(t.bna == 1).mean():.1%}/{(t.bna == 3).mean():.1%} of sub-blocks")
    for lab in ("bna", "lts_coarse"):
        print(f"{lab:11s} AUC any {s.loc[lab, 'auc_any_mean']:.3f}, top-10% capture {s.loc[lab, 'top10_capture_count_mean']:.3f}")
    print(f"fusion      AUC any {s.loc['fusion', 'auc_any_mean']:.3f}, top-10% capture {s.loc['fusion', 'top10_capture_count_mean']:.3f}")


if __name__ == "__main__":
    main()
