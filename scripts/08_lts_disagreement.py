"""Where RideScore LTS v1 and the out-of-fold crash model disagree (challenge 4.2 stretch goal).

Writes output/lts_disagreement.csv, output/lts_disagreement_map.png and output/risk_map.html.

Definitions (per DDOT sub-block, one consistent rule everywhere):
  LTS calm        = ref_lts <= 2      LTS stressful = ref_lts >= 3
  model high      = fusion pred_thresholds >= 1 (out-of-fold, leave-one-ward-out; the level-1 threshold is fitted
                    per fold, so about 13% of sub-blocks are "high")        model low = pred_thresholds == 0
Four cells: agree_low, agree_high, lts_calm_model_high, lts_stressful_model_low.
Wording: crashes are "recorded crashes", never a statement about danger or safety.

Usage: .venv/bin/python scripts/08_lts_disagreement.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.load_snapshot import load_snapshot, norm_key  # noqa: E402

CELLS = ["agree_low", "agree_high", "lts_calm_model_high", "lts_stressful_model_low"]
CELL_LABEL = {
    "agree_low": "LTS calm, model low (agree)",
    "agree_high": "LTS stressful, model high (agree)",
    "lts_calm_model_high": "LTS calm, but model high",
    "lts_stressful_model_low": "LTS stressful, but model low",
}
ORANGE, TEAL = "#e8480c", "#0f8b8d"
GREY, LIGHT = "#c9c9c9", "#ececec"
ATTR = "© OpenStreetMap contributors (ODbL); © CARTO; DDOT/MPD (CC BY 4.0)"


# ----------------------------------------------------------------------------- data
def build_frame(cfg):
    table = pd.read_parquet(cfg["paths"]["table"])
    table["dc_subblockkey"] = norm_key(table["dc_subblockkey"])
    oof = pd.read_parquet(cfg["paths"]["output_dir"] / "oof_predictions.parquet")
    oof = oof[oof["model"] == "fusion"][["dc_subblockkey", "mu", "pred_thresholds"]].copy()
    oof["dc_subblockkey"] = norm_key(oof["dc_subblockkey"])
    df = table.merge(oof, on="dc_subblockkey", how="left", validate="one_to_one")
    if df["mu"].isna().any():
        raise ValueError("sub-blocks without fusion OOF prediction")
    df["pred_thresholds"] = df["pred_thresholds"].astype(int)
    calm = df["ref_lts"] <= 2
    high = df["pred_thresholds"] >= 1
    df["cell"] = np.select(
        [calm & ~high, ~calm & high, calm & high, ~calm & ~high],
        ["agree_low", "agree_high", "lts_calm_model_high", "lts_stressful_model_low"])
    # NB: np.select order above is (calm, low)=agree_low, (stressful, high)=agree_high,
    # (calm, high)=calm-but-model-high, (stressful, low)=stressful-but-model-low
    df["any_bike_fac"] = (df["ddot_bike_best"].fillna(0) > 0) | (df["osmf_bike_best"].fillna(0) > 0)
    df["ddot_bike_fac"] = df["ddot_bike_best"].fillna(0) > 0
    return df


# ----------------------------------------------------------------------------- table
def top_fhwa(d, k=3):
    s = d["ddot_fhwa_class"].astype("object").where(d["ddot_fhwa_class"].notna(), "NA").astype(str)
    vc = s.str.replace(r"\.0$", "", regex=True).value_counts(normalize=True).head(k)
    return "; ".join(f"{i} ({v:.0%})" for i, v in vc.items())


def summarise(d, total_km, total_n):
    km = d["length_m"].sum() / 1000
    cr = int(d["crash_count"].sum())
    return {
        "n_subblocks": len(d), "share_of_subblocks": len(d) / total_n, "km": km, "share_of_network_km": km / total_km,
        "observed_crashes": cr, "share_of_all_crashes": np.nan,
        "crashes_per_km": cr / km if km else np.nan,
        "share_with_any_crash": (d["crash_count"] > 0).mean(), "share_level2": (d["level"] == 2).mean(),
        "top_fhwa_classes": top_fhwa(d),
        "share_any_bike_facility": d["any_bike_fac"].mean(), "share_ddot_bike_facility": d["ddot_bike_fac"].mean(),
        "share_lts1": (d["ref_lts"] == 1).mean(), "mean_model_score": d["mu"].mean(),
    }


def cell_table(df):
    tk, tn = df["length_m"].sum() / 1000, len(df)
    rows = [{"cell": c, "label": CELL_LABEL[c], **summarise(df[df["cell"] == c], tk, tn)} for c in CELLS]
    rows.append({"cell": "all", "label": "All sub-blocks", **summarise(df, tk, tn)})
    out = pd.DataFrame(rows)
    out["share_of_all_crashes"] = out["observed_crashes"] / df["crash_count"].sum()
    return out


# ----------------------------------------------------------------------------- segments
def segment_frame(cfg, snap, df):
    """Per-segment frame in snapshot order with cell / LTS / model attributes via dc_subblockkey."""
    out = cfg["paths"]["output_dir"]
    ex = pd.read_parquet(out / "crash_risk_by_segment.parquet")
    g = snap[["osm_u", "osm_v", "osm_key", "osm_name", "osm_highway", "dc_ROUTENAME", "dc_STREETNAME",
              "dc_subblockkey", "geometry"]].copy()
    g["dc_subblockkey"] = norm_key(g["dc_subblockkey"])
    chk = g[["osm_u", "osm_v", "osm_key"]].merge(ex, on=["osm_u", "osm_v", "osm_key"], how="left", validate="one_to_one")
    g["scored"] = chk["scored"].to_numpy()
    g["crash_risk_score"] = chk["crash_risk_score"].to_numpy()
    g["crash_risk_level"] = chk["crash_risk_level"].to_numpy()
    g["crash_count_5yr"] = chk["crash_count_5yr"].to_numpy()
    lut = df.set_index("dc_subblockkey")[["cell", "ref_lts"]]
    g = g.join(lut, on="dc_subblockkey")
    g["cell"] = g["cell"].where(g["scored"].fillna(False).astype(bool))
    g["street"] = (g["dc_ROUTENAME"].where(g["dc_ROUTENAME"].notna(), g["dc_STREETNAME"])
                   .where(lambda s: s.notna(), g["osm_name"]))
    return g


# ----------------------------------------------------------------------------- static map
def static_map(g, cfg, out):
    gm = g.to_crs(cfg["crs_metric"])
    cnt = gm["cell"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    gm[gm["cell"].isna()].plot(ax=ax, color=LIGHT, linewidth=0.3)
    gm[gm["cell"].isin(["agree_low", "agree_high"])].plot(ax=ax, color=GREY, linewidth=0.3)
    gm[gm["cell"] == "lts_stressful_model_low"].plot(ax=ax, color=TEAL, linewidth=1.0)
    gm[gm["cell"] == "lts_calm_model_high"].plot(ax=ax, color=ORANGE, linewidth=1.4)
    nseg = int(gm["cell"].notna().sum())
    ax.legend(handles=[
        Line2D([0], [0], color=ORANGE, lw=2.4,
               label=f"LTS calm (1-2), model high: {cnt.get('lts_calm_model_high', 0):,} segments"),
        Line2D([0], [0], color=TEAL, lw=2.0,
               label=f"LTS stressful (3-4), model low: {cnt.get('lts_stressful_model_low', 0):,} segments"),
        Line2D([0], [0], color=GREY, lw=1.2,
               label=f"Both agree: {int(cnt.get('agree_low', 0) + cnt.get('agree_high', 0)):,} segments"),
        Line2D([0], [0], color=LIGHT, lw=1.2, label=f"Unscored (no DDOT match): {len(gm) - nseg:,}")],
        loc="lower left", fontsize=8, frameon=True, title="Colour = recorded-crash model vs LTS v1", title_fontsize=8)
    ax.set_title("Where RideScore LTS and the crash model disagree\n"
                 "(model = out-of-fold crash-record level >= 1; leave-one-ward-out)", fontsize=10)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.savefig(out / "lts_disagreement_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- web map
def _round_coords(c, nd=5):
    if isinstance(c[0], (int, float)):
        return [round(c[0], nd), round(c[1], nd)]
    return [_round_coords(x, nd) for x in c]


def feature_collection(g, props_fn):
    from shapely.geometry import mapping
    feats = []
    for geom, props in zip(g.geometry.to_numpy(), props_fn(g)):
        if geom is None or geom.is_empty:
            continue
        m = mapping(geom)
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": m["type"], "coordinates": _round_coords(m["coordinates"])}})
    return {"type": "FeatureCollection", "features": feats}


def web_map(g, out):
    import branca.colormap as cm
    import folium
    from folium import JsCode

    gw = g.copy()
    gw["geometry"] = gw.geometry.simplify(0.00002)
    scored = gw["scored"].fillna(False).astype(bool)
    vmax = max(float(np.round(gw.loc[scored, "crash_risk_score"].quantile(0.99), 1)), 0.5)
    cmap = cm.LinearColormap(["#1a9850", "#ffffbf", "#d73027"], vmin=0, vmax=vmax,
                             caption=f"Model crash-risk score (expected recorded-crash level 0-2; colours capped at {vmax:g})")

    def colour(s):
        return "#cfcfcf" if pd.isna(s) else cmap(min(float(s), vmax))

    lts_txt = {1: "1 (calmest)", 2: "2", 3: "3", 4: "4 (most stressful)"}
    lvl_txt = {0: "0 (low)", 1: "1 (medium)", 2: "2 (high)"}

    def score_props(d):
        for r in d.itertuples():
            if bool(r.scored) and pd.notna(r.crash_risk_score):
                yield {"s": r.street if isinstance(r.street, str) else "unnamed",
                       "l": lts_txt.get(int(r.ref_lts), "n/a") if pd.notna(r.ref_lts) else "n/a",
                       "m": f"{lvl_txt[int(r.crash_risk_level)]}, score {r.crash_risk_score:.2f}",
                       "c": int(r.crash_count_5yr), "k": colour(r.crash_risk_score)}
            else:
                yield {"s": r.street if isinstance(r.street, str) else "unnamed", "l": "n/a",
                       "m": "not scored (no DDOT match)", "c": "n/a", "k": "#cfcfcf"}

    dis = gw[gw["cell"].isin(["lts_calm_model_high", "lts_stressful_model_low"])]
    dis_txt = {"lts_calm_model_high": "LTS calm, but model high",
               "lts_stressful_model_low": "LTS stressful, but model low"}
    dis_col = {"lts_calm_model_high": ORANGE, "lts_stressful_model_low": TEAL}

    def dis_props(d):
        for r in d.itertuples():
            yield {"s": r.street if isinstance(r.street, str) else "unnamed", "l": lts_txt[int(r.ref_lts)],
                   "m": f"{lvl_txt[int(r.crash_risk_level)]}, score {r.crash_risk_score:.2f}",
                   "c": int(r.crash_count_5yr), "d": dis_txt[r.cell], "k": dis_col[r.cell]}

    m = folium.Map(location=[38.9072, -77.0369], zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer("CartoDB positron", attr=ATTR, name="Base map (CartoDB positron)").add_to(m)

    style = JsCode("function(f){return {color: f.properties.k, weight: 2, opacity: 0.9};}")
    dstyle = JsCode("function(f){return {color: f.properties.k, weight: 3.5, opacity: 0.95};}")
    tip = folium.GeoJsonTooltip(
        fields=["s", "l", "m", "c"],
        aliases=["Street", "RideScore LTS v1", "Crash model (out-of-fold)", "Recorded bike crashes, 5 yr"], sticky=True)
    dtip = folium.GeoJsonTooltip(
        fields=["s", "d", "l", "m", "c"],
        aliases=["Street", "Disagreement", "RideScore LTS v1", "Crash model (out-of-fold)", "Recorded bike crashes, 5 yr"],
        sticky=True)

    fg1 = folium.FeatureGroup(name="Crash-model score by segment", show=True)
    folium.GeoJson(feature_collection(gw, lambda d: list(score_props(d))), style_function=style, tooltip=tip,
                   name="score").add_to(fg1)
    fg1.add_to(m)
    fg2 = folium.FeatureGroup(name="Where LTS and the crash model disagree", show=False)
    folium.GeoJson(feature_collection(dis, lambda d: list(dis_props(d))), style_function=dstyle, tooltip=dtip,
                   name="disagreement").add_to(fg2)
    fg2.add_to(m)
    cmap.add_to(m)

    n1 = int((dis["cell"] == "lts_calm_model_high").sum())
    n2 = int((dis["cell"] == "lts_stressful_model_low").sum())
    legend = f"""
    <div style="position: fixed; bottom: 24px; left: 12px; z-index: 9999; background: rgba(255,255,255,.93);
         padding: 8px 10px; font: 12px/1.35 system-ui, sans-serif; border: 1px solid #bbb; border-radius: 4px; max-width: 290px;">
      <b>Bike-crash record vs RideScore LTS, DC</b><br>
      Colour: out-of-fold model score from street design (green = low, red = high).<br>
      Toggle the layer control (top right) to see disagreements:<br>
      <span style="color:{ORANGE}"><b>&#9632;</b></span> LTS calm (1-2) but model high: {n1:,} segments<br>
      <span style="color:{TEAL}"><b>&#9632;</b></span> LTS stressful (3-4) but model low: {n2:,} segments<br>
      <span style="color:#999">Grey = no DDOT sub-block match.</span> Crashes are recorded bike crashes, not a measure of exposure.
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    path = out / "risk_map.html"
    m.save(str(path))
    return path


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    out = cfg["paths"]["output_dir"]
    df = build_frame(cfg)
    tab = cell_table(df)
    tab.round(4).to_csv(out / "lts_disagreement.csv", index=False)

    snap = load_snapshot(cfg)
    g = segment_frame(cfg, snap, df)
    static_map(g, cfg, out)
    path = web_map(g, out)

    print("definitions: LTS calm = ref_lts<=2, stressful >=3; model high = fusion OOF pred_thresholds>=1, low = 0")
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(tab[["cell", "n_subblocks", "km", "share_of_network_km", "observed_crashes", "share_of_all_crashes",
                   "crashes_per_km", "share_with_any_crash", "share_level2", "share_any_bike_facility",
                   "share_lts1"]].to_string(index=False))
        print(tab[["cell", "top_fhwa_classes"]].to_string(index=False))
    print("segments per cell:", g["cell"].value_counts(dropna=False).to_dict())
    print(f"risk_map.html: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
