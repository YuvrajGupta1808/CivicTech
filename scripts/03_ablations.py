"""E5 drop-one-group, E6 feature sources, E10 label variants, E11 drop weak matches (GBM + thresholds only).

Usage: python scripts/03_ablations.py [--only e5,e6,e10,e11] [--fast]
--fast trims cfg["gbm"]["fits"] to the first fit only (about 3x quicker); it is recorded in the `fits` column.
"""
import argparse
import contextlib
import copy
import io
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")  # many small fits in parallel beat few oversubscribed ones

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from src.config import load_config  # noqa: E402
from src.experiments import run_cv  # noqa: E402
from src.features.build import feature_columns  # noqa: E402
from src.features.groups import FEATURE_GROUPS, SOURCES, select_features  # noqa: E402

METRICS = ["macro_f1", "auc_any", "auc_repeat", "top10_capture_count"]
COLOUR = "#4c78a8"
SOURCE_SETS = {
    "ddot+net": ["ddot", "net"],
    "osm+net": ["osm", "net"],
    "ddot+osm": ["ddot", "osm"],
    "net only": ["net"],
    "all": ["ddot", "osm", "net"],
}


def cv_summary(table, feats, label_col, count_col, cfg, tag):
    """One GBM leave-one-ward-out run; returns a dict of metric means/sds across folds (thresholds decoder)."""
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        metrics, _ = run_cv(table, feats, label_col=label_col, cfg=cfg, models=("gbm",), count_col=count_col)
    m = metrics[(metrics["model"] == "gbm") & (metrics["decoder"] == "thresholds")]
    row = {"n_rows": len(table), "n_cols": len(feats), "n_folds": len(m),
           "fits": len(cfg["gbm"]["fits"]), "seconds": round(time.time() - t0, 1)}
    for k in METRICS:
        row[f"{k}_mean"] = m[k].mean()
        row[f"{k}_sd"] = m[k].std(ddof=1)
    row["_folds"] = m.set_index("test_ward")[METRICS]
    print(f"[abl] {tag:<28s} cols={len(feats):2d} rows={len(table):5d} "
          f"F1={row['macro_f1_mean']:.3f} AUCany={row['auc_any_mean']:.3f} "
          f"AUCrep={row['auc_repeat_mean']:.3f} top10={row['top10_capture_count_mean']:.3f} "
          f"({row['seconds']:.0f}s)", flush=True)
    return row


def paired_sd(rows, base, k):
    d = rows["_folds"][k] - base["_folds"][k]
    return d.std(ddof=1) if len(d.dropna()) > 1 else np.nan


def to_frame(rows, first_cols, base=None):
    recs = []
    for name, r in rows.items():
        rec = {first_cols: name, **{c: v for c, v in r.items() if c != "_folds"}}
        if base is not None:
            for k in METRICS:
                rec[f"delta_{k}"] = r[f"{k}_mean"] - base[f"{k}_mean"]
                rec[f"delta_{k}_sd"] = paired_sd(r, base, k)
        recs.append(rec)
    return pd.DataFrame(recs)


def show(df, cols, key):
    with pd.option_context("display.width", 220, "display.max_columns", 40,
                           "display.float_format", "{:.3f}".format):
        print(df[[key] + cols].to_string(index=False))


def bar_panel(ax, labels, vals, sds, title):
    y = np.arange(len(labels))
    ax.barh(y, vals, xerr=sds, color=COLOUR, error_kw=dict(ecolor="#9a9a9a", lw=0.8, capsize=2))
    ax.set_yticks(y, labels, fontsize=8)
    ax.axvline(0, color="grey", lw=1)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def plot_e5(df, out):
    d = df[df["group"] != "all"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, k, title in [(axes[0], "macro_f1", "Delta macro-F1 when group dropped"),
                         (axes[1], "auc_any", "Delta AUC(any crash) when group dropped")]:
        s = d.sort_values(f"delta_{k}", ascending=False)  # most negative (most important) at the bottom
        labels = [f"{g} ({n} left)" for g, n in zip(s["group"], s["n_cols"])]
        bar_panel(ax, labels[::-1], s[f"delta_{k}"].to_numpy()[::-1], s[f"delta_{k}_sd"].to_numpy()[::-1], title)
    fig.text(0.5, 0.005, "Bars: mean change vs all features; whiskers: sd of per-ward paired differences; "
             "negative = group matters", ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "ablation.png", dpi=160)
    plt.close(fig)


def plot_e6(df, out):
    d = df.sort_values("macro_f1_mean")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    for ax, k, title in [(axes[0], "macro_f1", "Macro-F1"), (axes[1], "auc_any", "AUC (any crash)")]:
        ax.barh(d["sources"], d[f"{k}_mean"], xerr=d[f"{k}_sd"], color=COLOUR,
                error_kw=dict(ecolor="#9a9a9a", lw=0.8, capsize=2))
        lo = (d[f"{k}_mean"] - d[f"{k}_sd"]).min()
        ax.set_xlim(max(0, lo - 0.05), None)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "sources.png", dpi=160)
    plt.close(fig)


def dedupe_blocks(table):
    b = table[table["dc_blockkey"].notna() & table["level_block"].notna() & table["crash_count_block"].notna()]
    b = b.sort_values("length_m", ascending=False).drop_duplicates("dc_blockkey", keep="first")
    b = b.sort_index().reset_index(drop=True)
    b["level_block"] = b["level_block"].astype(int)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="e5,e6,e10,e11")
    ap.add_argument("--fast", action="store_true", help="use only the first GBM fit from config")
    ap.add_argument("--jobs", type=int, default=6, help="parallel CV runs (each uses OMP_NUM_THREADS threads)")
    args = ap.parse_args()
    todo = {s.strip().lower() for s in args.only.split(",") if s.strip()}
    T0 = time.time()

    cfg = load_config()
    if args.fast:
        cfg = copy.deepcopy(cfg)
        cfg["gbm"]["fits"] = cfg["gbm"]["fits"][:1]
        print("[abl] --fast: cfg['gbm']['fits'] trimmed to the first fit only")
    out = ROOT / cfg["paths"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(cfg["paths"]["table"])
    feats = feature_columns(table)
    all_feats = select_features(feats)
    print(f"[abl] {len(table)} sub-blocks, {len(feats)} features ({len(all_feats)} in groups), "
          f"runs={sorted(todo)}, gbm fits={len(cfg['gbm']['fits'])}, jobs={args.jobs}, "
          f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

    # ---- plan every CV run: key -> (table, feature cols, label col, count col)
    jobs = {"base": (table, all_feats, "level", "crash_count")}
    present = {g: select_features(feats, groups=g) for g in FEATURE_GROUPS}
    if "e5" in todo:
        for g, cols_g in present.items():
            if cols_g:
                jobs[f"e5:{g}"] = (table, select_features(feats, drop_groups=g), "level", "crash_count")
    if "e6" in todo:
        for name, srcs in SOURCE_SETS.items():
            sel = select_features(feats, sources=srcs)
            if sel and name != "all":
                jobs[f"e6:{name}"] = (table, sel, "level", "crash_count")
    if "e10" in todo:
        jobs["e10:level_injury"] = (table, all_feats, "level_injury", "injury_count")
        jobs["e10:level_midblock"] = (table, all_feats, "level_midblock", "midblock_count")
        blocks = dedupe_blocks(table)
        print(f"[abl] block-level: {len(blocks)} blocks (longest sub-block's features; "
              f"{table['dc_blockkey'].isna().sum()} NaN-key sub-blocks dropped)")
        jobs["e10:level_block"] = (blocks, all_feats, "level_block", "crash_count_block")
    if "e11" in todo:
        keep = table[~table["weak_match"].astype(bool)]
        print(f"[abl] E11: dropping {len(table) - len(keep)} weak_match rows")
        jobs["e11:drop_weak"] = (keep, all_feats, "level", "crash_count")

    keys = list(jobs)
    res = Parallel(n_jobs=min(args.jobs, len(keys)))(
        delayed(cv_summary)(*jobs[k], cfg, k) for k in keys)
    R = dict(zip(keys, res))
    base = R["base"]
    cols = [f"{k}_{s}" for k in METRICS for s in ("mean", "sd")]
    dcols = [f"delta_{k}" for k in METRICS]

    if "e5" in todo:
        print("\n== E5 drop-one-group (delta vs all features) ==")
        rows = {"all": base}
        for g in present:
            if f"e5:{g}" in R:
                rows[g] = {**R[f"e5:{g}"], "n_dropped": len(present[g])}
        df = to_frame(rows, "group", base=base)
        df["n_dropped"] = df["n_dropped"].fillna(0).astype(int)
        df.drop(columns="seconds").to_csv(out / "ablation_groups.csv", index=False)
        show(df.sort_values("delta_macro_f1"), ["n_cols"] + cols + dcols, "group")
        plot_e5(df, out)

    if "e6" in todo:
        print("\n== E6 feature sources ==")
        rows = {n: (base if n == "all" else R[f"e6:{n}"]) for n in SOURCE_SETS
                if n == "all" or f"e6:{n}" in R}
        df = to_frame(rows, "sources", base=base)
        df.drop(columns="seconds").to_csv(out / "ablation_sources.csv", index=False)
        show(df, ["n_cols"] + cols + dcols, "sources")
        plot_e6(df, out)

    if "e10" in todo:
        print("\n== E10 label variants (all features; not comparable across rows: different targets) ==")
        rows = {"level (any, baseline)": base, "level_injury": R["e10:level_injury"],
                "level_midblock": R["e10:level_midblock"], "level_block": R["e10:level_block"]}
        df = to_frame(rows, "label")
        df["count_col"] = ["crash_count", "injury_count", "midblock_count", "crash_count_block"]
        df.drop(columns="seconds").to_csv(out / "label_variants.csv", index=False)
        show(df, ["n_rows"] + cols, "label")

    if "e11" in todo:
        print("\n== E11 drop weak_match ==")
        rows = {"all rows": base, "drop weak_match": R["e11:drop_weak"]}
        df = to_frame(rows, "variant", base=base)
        df.drop(columns="seconds").to_csv(out / "robustness.csv", index=False)
        show(df, ["n_rows"] + cols + dcols, "variant")

    print(f"\n[abl] done in {time.time() - T0:.0f}s (sum of per-run seconds "
          f"{sum(r['seconds'] for r in res):.0f}s)")


if __name__ == "__main__":
    main()
