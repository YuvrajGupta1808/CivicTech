"""BNA segment stress (stress 1 = comfortable, 3 = uncomfortable) per DDOT sub-block.

Source: the real `bikescore-bna` package (v0.2.0, git 6a6cb494, https://github.com/bright-fakl/bikescore-bna).
We call its packaged rule table `bikescore_bna.rules.stress_segment.default_segment_stress_rules()`
(`rules/data/segment_stress.yaml`: ordered first-match-wins rules over adj_fc, bike infra, speed, lanes, width)
directly on a DataFrame; nothing is reimplemented and no network graph or database is needed.
Intersection stress is NOT used (needs the node graph); this is segment stress only.

Inputs are built the way BNA's attributes stage builds them (standard-bna.yaml), from the sub-block table:
- functional class: `osmf_highway` (road classes), else the DDOT FHWA class (1 motorway, 2 trunk, 3 primary,
  4 secondary, 5-6 tertiary, 7 residential); cycleway/footway/path map to `path` only when source="osm".
  BNA's promotion step is applied: residential/unclassified becomes tertiary when the facility is a lane/buffered
  lane/track, when observed per-direction lanes > 1, or when observed speed >= 30.
- speed_limit (mph): `ddot_speed_max`, then `osmf_speed`, then BNA's per-class default
  (motorway 55, trunk/primary/secondary 40, tertiary 30, residential/unclassified 25, living_street/path 15).
- lanes per travel direction: `ddot_lanes_total`, then `osmf_lanes`; 0/NaN is "unobserved" and gets BNA's
  class default (2 for primary and above, 1 otherwise). Two-way: ceil(total/2) as in OSM `lanes` parsing;
  one-way: total (the reverse direction is blocked and ignored).
- bike facility: `ddot_bike_best` 1/2/3 -> lane/buffered_lane/track (symmetric, i.e. assumed to serve both
  directions); 0 -> none. Shared lanes (sharrows) are not a BNA stress input. Facility width (ft) is
  `ddot_bike_width / ddot_bike_lanes` per lane, defaulting to BNA's 5 ft when absent.
- `bicycle` tag: `osmf_bicycle_access` ("unknown" treated as absent).
Sub-block stress is the worst (max) stress over the valid travel directions.

source="fused" (default) uses DDOT first, OSM as fallback, as above. source="osm" uses OSM-derived columns only
(osmf_*), which is closest to what BNA does natively; DDOT columns are ignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BNA_VERSION = "bikescore-bna 0.2.0 (git 6a6cb494)"

_ROAD_FC = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}
_PATH_FC = {"cycleway", "footway", "path", "pedestrian", "track"}
_FHWA_TO_FC = {"1": "motorway", "2": "trunk", "3": "primary", "4": "secondary", "5": "tertiary",
               "6": "tertiary", "7": "residential"}
_FC_SPEED = {"motorway": 55, "motorway_link": 55, "trunk": 40, "trunk_link": 40, "primary": 40, "primary_link": 40,
             "secondary": 40, "secondary_link": 40, "tertiary": 30, "tertiary_link": 30, "residential": 25,
             "unclassified": 25, "living_street": 15, "path": 15, "track": 15}
_MULTI_LANE_FC = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                  "secondary", "secondary_link"}
_INFRA = {1: "lane", 2: "buffered_lane", 3: "track"}
_DEFAULT_FACILITY_WIDTH_FT = 5.0


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def bna_inputs(sub: pd.DataFrame, source: str = "fused") -> pd.DataFrame:
    """BNA attribute columns (adj_fc, speed_limit, ft/tf lanes and bike infra, bicycle) for each sub-block."""
    if source not in ("fused", "osm"):
        raise ValueError("source must be 'fused' or 'osm'")
    fused = source == "fused"
    idx = sub.index

    # functional class
    hw = _col(sub, "osmf_highway").astype("object").where(lambda s: s.notna(), None)
    hw = pd.Series(hw.to_numpy(dtype=object), index=idx)
    fc = hw.where(hw.isin(_ROAD_FC))
    if fused:
        fhwa = _col(sub, "ddot_fhwa_class").astype("object").map(
            lambda v: _FHWA_TO_FC.get(str(v).split(".")[0]) if pd.notna(v) else None)
        fc = fc.where(fc.notna(), pd.Series(fhwa.to_numpy(dtype=object), index=idx))
    else:
        fc = fc.where(fc.notna(), hw.where(hw.isin(_PATH_FC)).map(lambda v: "track" if v == "track" else "path"))
    fc = fc.where(fc.notna(), "residential")

    # speed (mph); NaN or <=0 is unobserved
    speed = _num(_col(sub, "ddot_speed_max")) if fused else pd.Series(np.nan, index=idx)
    osm_speed = _num(_col(sub, "osmf_speed"))
    speed = speed.where(speed > 0, osm_speed.where(osm_speed > 0))
    obs_speed = speed.copy()
    speed = speed.fillna(fc.map(_FC_SPEED).astype(float))

    # lanes: total through lanes -> per direction
    total = _num(_col(sub, "ddot_lanes_total")) if fused else pd.Series(np.nan, index=idx)
    osm_total = _num(_col(sub, "osmf_lanes"))
    total = total.where(total > 0, osm_total.where(osm_total > 0))
    oneway_col = "ddot_oneway" if fused else "osmf_oneway"
    oneway = _num(_col(sub, oneway_col)).fillna(0).eq(1)
    if fused:  # DDOT oneway is NaN where no lanes are recorded; fall back to OSM there
        oneway = oneway | (_num(_col(sub, "ddot_oneway")).isna() & _num(_col(sub, "osmf_oneway")).eq(1))
    per_dir_obs = pd.Series(np.where(oneway, total, np.ceil(total / 2.0)), index=idx)
    obs_multi = per_dir_obs > 1
    default_lanes = fc.map(lambda v: 2.0 if v in _MULTI_LANE_FC else 1.0)
    per_dir = per_dir_obs.fillna(default_lanes)

    # bike facility
    best = _num(_col(sub, "ddot_bike_best")) if fused else _num(_col(sub, "osmf_bike_best"))
    best = best.fillna(0).round().astype(int)  # osmf 0.5 (shared lane) rounds to 0 (half-to-even): not a BNA input
    infra = best.map(_INFRA)
    width_ft = pd.Series(np.nan, index=idx)
    if fused:
        w = _num(_col(sub, "ddot_bike_width"))
        n = _num(_col(sub, "ddot_bike_lanes")).clip(lower=1)
        width_ft = (w / n).where(w > 0)
    width_ft = width_ft.fillna(_DEFAULT_FACILITY_WIDTH_FT)

    # BNA class promotion (residential/unclassified -> tertiary) on observed attributes
    both_lane = best.isin([1, 2, 3])
    promote = fc.isin(["residential", "unclassified"]) & (both_lane | obs_multi | (obs_speed >= 30))
    fc = fc.where(~promote, "tertiary")

    bicycle = _col(sub, "osmf_bicycle_access").astype("object").where(lambda s: s.notna() & (s != "unknown"), None)
    infra_o = pd.Series(infra.to_numpy(dtype=object), index=idx)
    return pd.DataFrame({
        "adj_fc": fc, "bicycle": pd.Series(bicycle.to_numpy(dtype=object), index=idx), "speed_limit": speed,
        "ft_lanes": per_dir, "tf_lanes": per_dir.where(~oneway, 0.0),
        "ft_bike_infra": infra_o, "tf_bike_infra": infra_o,
        "ft_bike_infra_width": width_ft, "tf_bike_infra_width": width_ft,
        "one_way": oneway,
    }, index=idx)


def bna_stress(sub: pd.DataFrame, source: str = "fused") -> pd.Series:
    """Return BNA segment stress (1 comfortable / 3 uncomfortable) per sub-block, indexed by dc_subblockkey."""
    from bikescore_bna.rules.stress_segment import default_segment_stress_rules

    inp = bna_inputs(sub, source)
    key = sub["dc_subblockkey"].to_numpy() if "dc_subblockkey" in sub.columns else sub.index.to_numpy()
    inp = inp.reset_index(drop=True)
    out = default_segment_stress_rules().apply(inp)
    ft = pd.to_numeric(out["ft_seg_stress"], errors="coerce")
    tf = pd.to_numeric(out["tf_seg_stress"], errors="coerce").where(~inp["one_way"])  # BNA one-way reset
    stress = pd.concat([ft, tf], axis=1).max(axis=1, skipna=True)  # worst valid direction
    if stress.isna().any():
        raise ValueError(f"BNA left {int(stress.isna().sum())} sub-blocks without a stress value")
    stress = stress.astype(int)
    stress.index = pd.Index(key, name="dc_subblockkey")
    stress.name = "bna_stress"
    return stress
