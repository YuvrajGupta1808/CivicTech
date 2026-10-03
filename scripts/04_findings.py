"""Findings: surprise groups (OOF), LTS vs crashes (E2), maps, and the hand-in Parquet.

Usage: .venv/bin/python scripts/04_findings.py [--standin]
--standin fakes mu from ddot_log_aadt when output/oof_predictions.parquet does not exist yet (development only).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import geopandas as gpd  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.fetch_crashes import fetch_crashes  # noqa: E402
from src.data.load_snapshot import load_snapshot, norm_key  # noqa: E402
from src.export import OUT_NAME, export_segments  # noqa: E402

GREY, RED, BLUE = "#d4d4d4", "#d62728", "#1f5fbf"
G_SAFE, G_RISKY = "looks_safe_has_crashes", "looks_risky_no_crashes"
GROUP_ORDER = ["all", "level2", "pred2", G_SAFE, G_RISKY]


# ----------------------------------------------------------------------------- inputs
def standin_oof(table: pd.DataFrame) -> pd.DataFrame:
    """Development stand-in: expected level from ddot_log_aadt rank + noise; thresholds at the 90th/97th pct."""
    rng = np.random.default_rng(0)
    a = table["ddot_log_aadt"].fillna(table["ddot_log_aadt"].median())
    r = pd.Series(a.rank(pct=True).to_numpy() + rng.normal(0, 0.15, len(a))).rank(pct=True).to_numpy()
    mu = 2 * r ** 4
    t1, t2 = np.quantile(mu, [0.90, 0.97])
    pred = (mu > t1).astype(int) + (mu > t2).astype(int)
    p2 = np.clip(mu / 2, 0, 1) ** 2
    return pd.DataFrame({
        "dc_subblockkey": table["dc_subblockkey"].to_numpy(), "ward": table["ward"].to_numpy(),
        "model": "fusion", "p0": 1 - np.clip(mu / 2, 0, 1), "p1": np.clip(mu / 2, 0, 1) - p2, "p2": p2,
        "mu": mu, "pred_argmax": pd.array(pred, dtype="Int64"), "pred_bias": pd.array(pred, dtype="Int64"),
        "pred_thresholds": pd.array(pred, dtype="Int64"), "thr1": t1, "thr2": t2})


def load_main_oof(cfg, table, standin):
    path = cfg["paths"]["output_dir"] / "oof_predictions.parquet"
    if path.exists():
        oof = pd.read_parquet(path)
        print(f"[findings] OOF file: {path} ({len(oof)} rows, models {sorted(oof['model'].unique())})")
    elif standin:
        print("[findings] !!! STAND-IN OOF (fake mu from ddot_log_aadt); numbers are NOT results !!!")
        oof = standin_oof(table)
    else:
        raise FileNotFoundError(f"{path} missing; run scripts/02_ladder.py (or pass --standin to develop)")
    main = "fusion" if "fusion" in set(oof["model"]) else "gbm"
    oof = oof[oof["model"] == main].copy()
    oof["dc_subblockkey"] = norm_key(oof["dc_subblockkey"])
    if oof["dc_subblockkey"].duplicated().any():
        raise ValueError("more than one OOF row per sub-block for the main model")
    print(f"[findings] main model = {main}: {len(oof)} sub-blocks")
    return oof, main


def join_table(table, oof):
    t = table.copy()
    t["dc_subblockkey"] = norm_key(t["dc_subblockkey"])
    keep = ["dc_subblockkey", "ward", "model", "p0", "p1", "p2", "mu", "pred_thresholds"]
    df = t.merge(oof[keep].rename(columns={"ward": "oof_ward"}), on="dc_subblockkey", how="left",
                 validate="one_to_one")
    if df["mu"].isna().any():
        raise ValueError(f"{int(df['mu'].isna().sum())} sub-blocks have no OOF prediction")
    if (df["ward"] != df["oof_ward"]).any():
        raise ValueError("ward mismatch between table and OOF")
    df["pred_thresholds"] = df["pred_thresholds"].astype(int)
    return df.drop(columns="oof_ward")


# ----------------------------------------------------------------------------- surprise groups
def surprise_masks(df):
    med = float(df["mu"].median())
    safe = (df["level"] == 2) & ((df["pred_thresholds"] == 0) | (df["mu"] < med))
    risky = (df["pred_thresholds"] == 2) & (df["level"] == 0)
    masks = {"all": pd.Series(True, index=df.index), "level2": df["level"] == 2,
             "pred2": df["pred_thresholds"] == 2, G_SAFE: safe, G_RISKY: risky}
    return masks, med


def _row(metric, value, series_by_group):
    return {"metric": metric, "value": value, **series_by_group}


def breakdown(df, masks):
    rows = []

    def add(metric, value, fn):
        rows.append(_row(metric, value, {g: fn(df[m]) for g, m in masks.items()}))

    add("n_subblocks", "", len)
    add("mean_mu", "", lambda d: d["mu"].mean())
    add("mean_length_m", "", lambda d: d["length_m"].mean())
    add("crashes_total", "", lambda d: int(d["crash_count"].sum()))

    def int_share(d):
        d = d[d["crash_count"] > 0]
        return (d["int_count"] / d["crash_count"]).mean() if len(d) else np.nan
    add("mean_intersection_share_of_crashes", "sub-blocks with >=1 crash", int_share)
    add("pooled_intersection_share_of_crashes", "sum int / sum crashes",
        lambda d: d["int_count"].sum() / d["crash_count"].sum() if d["crash_count"].sum() else np.nan)
    add("mean_midblock_share_of_crashes", "sub-blocks with >=1 crash",
        lambda d: (d.loc[d["crash_count"] > 0, "midblock_count"] / d.loc[d["crash_count"] > 0, "crash_count"]).mean()
        if (d["crash_count"] > 0).any() else np.nan)

    legs = df["net_max_legs"]
    add("net_max_legs", "share >= 4", lambda d: (d["net_max_legs"] >= 4).mean())
    add("net_max_legs", "mean", lambda d: d["net_max_legs"].mean())
    for lab, f in [("share <= 2", lambda s: s <= 2), ("share == 3", lambda s: s == 3),
                   ("share == 4", lambda s: s == 4), ("share >= 5", lambda s: s >= 5)]:
        add("net_max_legs", lab, lambda d, f=f: f(d["net_max_legs"]).mean())
    del legs

    def cat_shares(col, prefix):
        s = df[col].astype("object").where(df[col].notna(), "missing").astype(str)
        for v in sorted(s.unique(), key=lambda x: (x == "missing", x)):
            add(f"{prefix}_share", v, lambda d, v=v: (
                d[col].astype("object").where(d[col].notna(), "missing").astype(str) == v).mean())

    cat_shares("ddot_fhwa_class", "ddot_fhwa_class")
    cat_shares("osmf_highway", "osmf_highway")
    cat_shares("ddot_bike_best", "ddot_bike_best")
    cat_shares("osmf_bike_best", "osmf_bike_best")
    add("ddot_bike_best", "share > 0 (any facility)", lambda d: (d["ddot_bike_best"] > 0).mean())
    add("osmf_bike_best", "share > 0 (any facility)", lambda d: (d["osmf_bike_best"] > 0).mean())
    for w in sorted(df["ward"].unique()):
        add("ward_count", f"Ward {w}", lambda d, w=w: int((d["ward"] == w).sum()))
    for w in sorted(df["ward"].unique()):
        add("ward_share", f"Ward {w}", lambda d, w=w: (d["ward"] == w).mean())
    add("weak_match_share", "", lambda d: d["weak_match"].mean())
    for k in sorted(df["ref_lts"].unique()):
        add("ref_lts_share", f"LTS {k}", lambda d, k=k: (d["ref_lts"] == k).mean())
    return pd.DataFrame(rows)[["metric", "value"] + GROUP_ORDER]


def patterns(bd, group, ref="all", n=5):
    """Largest absolute share differences (percentage points) of the group versus a reference group."""
    is_share = bd["metric"].str.endswith("_share") | bd["value"].str.startswith("share")
    share = bd[is_share & ~bd["metric"].str.startswith(("mean", "pooled"))].copy()
    share["diff_pp"] = 100 * (share[group] - share[ref])
    share["lift"] = share[group] / share[ref].replace(0, np.nan)
    share = share[share[ref] >= 0.02]
    return share.reindex(share["diff_pp"].abs().sort_values(ascending=False).index).head(n)


def surprise_examples(df, snap, masks, med, crashes, n=15):
    first = snap.dropna(subset=["dc_subblockkey"]).drop_duplicates("dc_subblockkey").set_index("dc_subblockkey")
    names = pd.DataFrame({
        "street": first["dc_ROUTENAME"].where(first["dc_ROUTENAME"].notna(), first["dc_STREETNAME"]),
        "from_street": first["dc_FROMSTREET"], "to_street": first["dc_TOSTREET"], "osm_name": first["osm_name"]})
    near = (crashes.dropna(subset=["subblockkey", "NEARESTINTSTREETNAME"])
            .groupby("subblockkey")["NEARESTINTSTREETNAME"]
            .agg(lambda x: " / ".join(x.value_counts().index[:2])).rename("nearest_int_streets"))
    d = df[masks[G_SAFE]].sort_values(["crash_count", "mu"], ascending=[False, True]).head(n)
    d = d.join(names, on="dc_subblockkey").join(near, on="dc_subblockkey")
    d["why_looks_safe"] = np.where(d["pred_thresholds"] == 0, "predicted level 0", "score below median")
    cols = ["street", "from_street", "to_street", "nearest_int_streets", "ward", "crash_count", "int_count", "midblock_count",
            "injury_count", "mu", "pred_thresholds", "why_looks_safe", "ref_lts", "net_max_legs", "length_m",
            "weak_match", "osm_name", "dc_subblockkey"]
    out = d[cols].reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# ----------------------------------------------------------------------------- LTS vs crashes (E2)
def lts_vs_crashes(df):
    ct = pd.crosstab(df["ref_lts"], df["level"], normalize="index").add_prefix("share_level_")
    g = df.groupby("ref_lts")
    out = pd.DataFrame({
        "n_subblocks": g.size(), "length_km": g["length_m"].sum() / 1000, "crashes": g["crash_count"].sum(),
        "mean_model_score": g["mu"].mean(), "share_predicted_level2": g["pred_thresholds"].apply(lambda s: (s == 2).mean())})
    out["crashes_per_km"] = out["crashes"] / out["length_km"]
    out["share_any_crash"] = 1 - ct["share_level_0"]
    out["share_of_all_crashes"] = out["crashes"] / out["crashes"].sum()
    out["share_of_network_km"] = out["length_km"] / out["length_km"].sum()
    out = out.join(ct)
    tot = pd.Series({
        "n_subblocks": len(df), "length_km": df["length_m"].sum() / 1000, "crashes": df["crash_count"].sum(),
        "mean_model_score": df["mu"].mean(), "share_predicted_level2": (df["pred_thresholds"] == 2).mean(),
        "crashes_per_km": df["crash_count"].sum() / (df["length_m"].sum() / 1000),
        "share_any_crash": (df["level"] > 0).mean(), "share_of_all_crashes": 1.0, "share_of_network_km": 1.0,
        **{f"share_level_{k}": (df["level"] == k).mean() for k in (0, 1, 2)}}, name="all")
    out = pd.concat([out, tot.to_frame().T])
    out.index = [f"LTS {i}" if i != "all" else "all" for i in out.index]
    out.index.name = "ref_lts"
    return out.reset_index()


# ----------------------------------------------------------------------------- maps
def _style(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")


def risk_map(g, out):
    fig, ax = plt.subplots(figsize=(8, 8))
    g[~g["scored"]].plot(ax=ax, color=GREY, linewidth=0.3)
    sc = g[g["scored"]].sort_values("crash_risk_score")
    vmax = max(float(np.round(sc["crash_risk_score"].quantile(0.99), 1)), 0.5)  # score is skewed: cap at p99
    sc.plot(ax=ax, column="crash_risk_score", cmap="RdYlGn_r", vmin=0, vmax=vmax, linewidth=0.6, legend=True,
            legend_kwds={"shrink": 0.45, "label": f"Out-of-fold crash-risk score (expected level 0-2; colours capped at {vmax:g})"})
    ax.set_title("Where bike crashes are expected: out-of-fold score by segment, DC\n"
                 "(grey = no DDOT sub-block match, unscored)", fontsize=10)
    _style(ax)
    fig.savefig(out / "risk_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def surprise_map(g, masks_by_key, out, n_blocks):
    safe = g["dc_subblockkey"].isin(masks_by_key[G_SAFE])
    risky = g["dc_subblockkey"].isin(masks_by_key[G_RISKY])
    fig, ax = plt.subplots(figsize=(8, 8))
    g[~(safe | risky)].plot(ax=ax, color=GREY, linewidth=0.3)
    g[risky].plot(ax=ax, color=BLUE, linewidth=1.2)
    g[safe].plot(ax=ax, color=RED, linewidth=1.6)
    ax.legend(handles=[
        Line2D([0], [0], color=RED, lw=2.4, label=f"Looks safe, has crashes ({n_blocks[G_SAFE]} sub-blocks)"),
        Line2D([0], [0], color=BLUE, lw=2.0, label=f"Looks risky, no crashes ({n_blocks[G_RISKY]} sub-blocks)"),
        Line2D([0], [0], color=GREY, lw=1.2, label="Model and crash record agree / other")],
        loc="lower left", fontsize=8, frameon=True)
    ax.set_title("Where the model and the crash record disagree (out-of-fold, leave-one-ward-out)", fontsize=10)
    _style(ax)
    fig.savefig(out / "surprise_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def crash_points_map(g, crashes, cfg, out):
    c = crashes.dropna(subset=["LATITUDE", "LONGITUDE"])
    c = c[(c["LATITUDE"] != 0) & (c["LONGITUDE"] != 0)]
    pts = gpd.GeoSeries(gpd.points_from_xy(c["LONGITUDE"], c["LATITUDE"]), crs="EPSG:4326").to_crs(cfg["crs_metric"])
    fig, ax = plt.subplots(figsize=(8, 8))
    g.plot(ax=ax, color=GREY, linewidth=0.3)
    inj = c["injured"].to_numpy()
    pts[~inj].plot(ax=ax, color="#ff7f0e", markersize=4, alpha=0.5, linewidth=0)
    pts[inj].plot(ax=ax, color=RED, markersize=5, alpha=0.7, linewidth=0)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", label="crash, no cyclist injury recorded"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, label="cyclist injured")],
              loc="lower left", fontsize=8)
    ax.set_title(f"Bike-involved crashes, {cfg['crashes']['window_start']} to {cfg['snapshot_date']} (n={len(c)})",
                 fontsize=10)
    _style(ax)
    fig.savefig(out / "crash_points.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- verification
def verify(cfg, snap):
    path = cfg["paths"]["output_dir"] / OUT_NAME
    ex = pd.read_parquet(path)
    meta = pq.read_schema(path).metadata
    scored = ex["scored"]
    print("\n[verify] hand-in:", path)
    print(f"  rows: {len(ex)} (snapshot {len(snap)}); unique (osm_u, osm_v, osm_key): "
          f"{not ex.duplicated(['osm_u', 'osm_v', 'osm_key']).any()}")
    print(f"  scored: {int(scored.sum())} (expect 27,579); unscored: {int((~scored).sum())}; "
          f"scored==False exactly for match_status=='none': "
          f"{bool((scored.to_numpy() == (snap['match_status'].to_numpy() != 'none')).all())}")
    print("  level distribution (scored):", {int(k): int(v) for k, v in ex.loc[scored, "crash_risk_level"].value_counts().sort_index().items()},
          "| level NA when unscored:", bool(ex.loc[~scored, "crash_risk_level"].isna().all()))
    s = ex.loc[scored, "crash_risk_score"]
    print(f"  score range: [{s.min():.4f}, {s.max():.4f}] mean {s.mean():.4f}; NaN when unscored: "
          f"{bool(ex.loc[~scored, 'crash_risk_score'].isna().all())}")
    print("  dtypes:", {k: str(v) for k, v in ex.dtypes.items()})
    print("  metadata:", {k.decode(): v.decode()[:80] for k, v in meta.items() if k != b"pandas"})


def main():
    standin = "--standin" in sys.argv
    cfg = load_config()
    out = cfg["paths"]["output_dir"]
    table = pd.read_parquet(cfg["paths"]["table"])
    snap = load_snapshot(cfg)
    oof, main_model = load_main_oof(cfg, table, standin)
    df = join_table(table, oof)

    # --- surprise groups
    masks, med = surprise_masks(df)
    bd = breakdown(df, masks)
    bd.round(4).to_csv(out / "surprise_breakdown.csv", index=False)
    crashes = fetch_crashes(cfg)
    ex = surprise_examples(df, snap, masks, med, crashes)
    ex.round(4).to_csv(out / "surprise_examples.csv", index=False)

    # --- LTS vs crashes (E2)
    lts = lts_vs_crashes(df)
    lts.round(4).to_csv(out / "lts_vs_crashes.csv", index=False)

    # --- maps
    g = snap[["osm_u", "osm_v", "osm_key", "match_status", "dc_subblockkey", "geometry"]].copy()
    g["dc_subblockkey"] = norm_key(g["dc_subblockkey"])
    exp = export_segments(snap, oof, table, cfg)  # writes the hand-in Parquet
    g = g.to_crs(cfg["crs_metric"])
    g["crash_risk_score"] = exp["crash_risk_score"].to_numpy()
    g["scored"] = exp["scored"].to_numpy()
    keys = {G_SAFE: set(df.loc[masks[G_SAFE], "dc_subblockkey"]), G_RISKY: set(df.loc[masks[G_RISKY], "dc_subblockkey"])}
    risk_map(g, out)
    surprise_map(g, keys, out, {G_SAFE: len(keys[G_SAFE]), G_RISKY: len(keys[G_RISKY])})
    crash_points_map(g, crashes, cfg, out)

    verify(cfg, snap)

    # --- compact summary
    n = {k: int(m.sum()) for k, m in masks.items()}
    safe_a = int(((df["level"] == 2) & (df["pred_thresholds"] == 0)).sum())
    safe_b = int(((df["level"] == 2) & (df["mu"] < med)).sum())
    print(f"\n[summary] main model: {main_model}{' (STAND-IN)' if standin and not (out / 'oof_predictions.parquet').exists() else ''}")
    print(f"  sub-blocks {len(df)}, level-2 {n['level2']}, predicted level 2 {n['pred2']}, median mu {med:.4f}")
    print(f"  looks safe, has crashes: {n[G_SAFE]} sub-blocks ({n[G_SAFE] / max(n['level2'], 1):.1%} of level-2); "
          f"predicted 0: {safe_a}, mu < median: {safe_b}; crashes {int(df.loc[masks[G_SAFE], 'crash_count'].sum())}")
    print(f"  looks risky, no crashes: {n[G_RISKY]} sub-blocks ({n[G_RISKY] / max(n['pred2'], 1):.1%} of predicted-2)")
    b = bd.set_index(["metric", "value"])
    for lab, key in [("intersection share (mean)", ("mean_intersection_share_of_crashes", "sub-blocks with >=1 crash")),
                     ("net_max_legs>=4", ("net_max_legs", "share >= 4")),
                     ("weak_match", ("weak_match_share", "")),
                     ("any ddot bike facility", ("ddot_bike_best", "share > 0 (any facility)")),
                     ("any osm bike facility", ("osmf_bike_best", "share > 0 (any facility)"))]:
        r = b.loc[key]
        print(f"  {lab}: all {r['all']:.3f} | level2 {r['level2']:.3f} | safe-has-crashes {r[G_SAFE]:.3f} | "
              f"risky-no-crashes {r[G_RISKY]:.3f}")
    for grp, ref in [(G_SAFE, "all"), (G_SAFE, "level2"), (G_RISKY, "all"), (G_RISKY, "pred2")]:
        print(f"  top patterns, {grp} vs {ref} sub-blocks (percentage-point difference):")
        for _, r in patterns(bd, grp, ref).iterrows():
            print(f"    {r['metric']}={r['value']}: group {r[grp]:.3f} vs {ref} {r[ref]:.3f} ({r['diff_pp']:+.1f} pp)")
    print("  LTS vs crashes:")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(lts[["ref_lts", "n_subblocks", "length_km", "crashes", "crashes_per_km", "share_any_crash",
                   "share_of_all_crashes", "share_of_network_km", "share_predicted_level2"]].to_string(index=False))
    print("  top-15 examples: output/surprise_examples.csv")
    print(ex[["street", "nearest_int_streets", "ward", "crash_count", "mu"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
