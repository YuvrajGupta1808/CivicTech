"""OSM-derived features per DDOT sub-block (prefix osmf_).

Pure text parsers (unit-tested) plus `osm_features`, which aggregates the
OSM segments matched to each sub-block into one feature row.
"""
import re

import numpy as np
import pandas as pd

KM_PER_MILE = 1.609344
FT_TO_M = 0.3048
IN_TO_M = 0.0254

FACILITY_RANK = {"track": 3, "buffered": 2, "lane": 1, "shared": 0.5, "none": 0}

# Non-road ways: separately mapped bike/foot infrastructure and service tracks.
NON_ROAD_HIGHWAYS = {
    "cycleway", "path", "footway", "pedestrian", "steps", "bridleway", "track",
}

_TRACK = {"track", "separate", "opposite_track"}
_LANE = {"lane", "opposite_lane", "exclusive", "exclusive_lane"}
_SHARED = {"shared_lane", "share_busway", "shared_busway", "shoulder", "shared"}
_NUM = re.compile(r"\d+(?:\.\d+)?")
_SPEED = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?\s*(?:[;|].*)?$", re.I)
_FT_IN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*'\s*(?:(\d+(?:\.\d+)?)\s*\"?)?\s*$")
_WIDTH = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(m|ft)?\s*$", re.I)

SIDE_FIELDS = [
    ("osm_cycleway", "osm_cycleway_buffer"),
    ("osm_cycleway_left", "osm_cycleway_left_buffer"),
    ("osm_cycleway_right", "osm_cycleway_right_buffer"),
    ("osm_cycleway_both", "osm_cycleway_both_buffer"),
]
# *_restriction columns hold no_stopping / no_parking, which is not parking supply.
PARKING_COLS = [
    "osm_parking_left", "osm_parking_right", "osm_parking_both",
    "osm_parking_lane_left", "osm_parking_lane_right",
]
OUT_COLUMNS = [
    "osmf_highway", "osmf_speed", "osmf_lanes", "osmf_width", "osmf_oneway",
    "osmf_contraflow", "osmf_bike_best", "osmf_bike_both_sides",
    "osmf_bike_buffered", "osmf_bicycle_access", "osmf_has_parallel_track",
    "osmf_parking_any", "osmf_turn_lanes", "osmf_turn_left", "osmf_turn_right",
    "osmf_n_segments",
]


# ---------------------------------------------------------------- parsers
def parse_maxspeed(s) -> float:
    """OSM maxspeed text -> mph. No unit means mph; km/h is converted."""
    if s is None or pd.isna(s):
        return np.nan
    m = _SPEED.match(str(s))
    if not m:
        return np.nan  # 'none', 'signals', 'walk', junk
    val = float(m.group(1))
    unit = (m.group(2) or "mph").lower()
    return val / KM_PER_MILE if unit != "mph" else val


def parse_lanes(s) -> float:
    """Lane count from text; '2;3' -> 3 (max of the integers found)."""
    if s is None or pd.isna(s):
        return np.nan
    nums = re.findall(r"\d+", str(s))
    return float(max(int(n) for n in nums)) if nums else np.nan


def parse_turn_lanes(s) -> tuple:
    """turn:lanes text -> (n_lanes, has_left, has_right); NaN, 0, 0 if missing."""
    if s is None or pd.isna(s) or not str(s).strip():
        return (np.nan, 0, 0)
    text = str(s).strip().lower()
    return (float(text.count("|") + 1), int("left" in text), int("right" in text))


def parse_width(s) -> float:
    """OSM width text -> metres. Plain numbers are metres; 31'0\" is feet/inches."""
    if s is None or pd.isna(s):
        return np.nan
    text = str(s).strip()
    m = _FT_IN.match(text)
    if m:
        return float(m.group(1)) * FT_TO_M + float(m.group(2) or 0) * IN_TO_M
    m = _WIDTH.match(text)
    if m:
        val = float(m.group(1))
        return val * FT_TO_M if (m.group(2) or "").lower() == "ft" else val
    return np.nan


def _is_buffered(buffer) -> bool:
    """Buffer tag is 'yes' or a positive number (optionally with a unit)."""
    if buffer is None or pd.isna(buffer):
        return False
    text = str(buffer).strip().lower()
    if text in {"yes", "true"}:
        return True
    m = _NUM.match(text)
    return bool(m) and float(m.group(0)) > 0


def normalise_cycleway(value, buffer=None) -> str:
    """Map an OSM cycleway tag (+ buffer tag) to track/buffered/lane/shared/none."""
    if value is None or pd.isna(value):
        return "none"
    best = "none"
    for tok in str(value).lower().replace("|", ";").split(";"):
        tok = tok.strip()
        if tok in _TRACK:
            cls = "track"
        elif tok == "buffered_lane":
            cls = "buffered"
        elif tok in _LANE:
            cls = "buffered" if _is_buffered(buffer) else "lane"
        elif tok in _SHARED:
            cls = "shared"
        else:
            cls = "none"  # no, none, crossing, traffic_island, sidewalk, junk
        if FACILITY_RANK[cls] > FACILITY_RANK[best]:
            best = cls
    return best


# ------------------------------------------------------------ aggregation
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column as a Series, or all-NaN when the snapshot lacks it."""
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index, dtype="object")


def _lower(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def _weighted_mode(df: pd.DataFrame, key: str, col: str, weight: str) -> pd.Series:
    """Per-key value with the largest summed weight; ties go to the first alphabetically."""
    agg = df.groupby([key, col], observed=True)[weight].sum().reset_index()
    agg = agg.sort_values([key, weight, col], ascending=[True, False, True])
    return agg.drop_duplicates(key).set_index(key)[col]


def _segment_table(seg: pd.DataFrame) -> pd.DataFrame:
    """One parsed row per segment (all columns still segment-level)."""
    t = pd.DataFrame(index=seg.index)
    t["key"] = seg["dc_subblockkey"]
    t["w"] = seg["seg_length_m"].fillna(0.0) if "seg_length_m" in seg else 1.0
    t["highway"] = _lower(_col(seg, "osm_highway")).fillna("unknown")
    t["is_road"] = ~t["highway"].isin(NON_ROAD_HIGHWAYS)
    t["speed"] = _col(seg, "osm_maxspeed").map(parse_maxspeed).astype(float)
    t["lanes"] = _col(seg, "osm_lanes").map(parse_lanes).astype(float)
    t["width"] = _col(seg, "osm_width").map(parse_width).astype(float)

    oneway = _lower(_col(seg, "osm_oneway")).isin(["yes", "true", "1", "-1"])
    t["oneway"] = oneway.astype(int)
    bike_oneway = _lower(_col(seg, "osm_oneway_bicycle")) == "no"
    t["contraflow"] = (oneway & bike_oneway.fillna(False)).astype(int)

    # Cycleway classes per side field, with each side's own buffer tag.
    classes = {}
    for field, buf in SIDE_FIELDS:
        vals, bufs = _col(seg, field), _col(seg, buf)
        classes[field] = pd.Series(
            [normalise_cycleway(v, b) for v, b in zip(vals, bufs)], index=seg.index
        )
    ranks = {f: c.map(FACILITY_RANK).astype(float) for f, c in classes.items()}
    t["bike_best"] = pd.concat(ranks.values(), axis=1).max(axis=1)
    left, right = ranks["osm_cycleway_left"], ranks["osm_cycleway_right"]
    t["bike_both"] = (((left >= 1) & (right >= 1)) | (ranks["osm_cycleway_both"] >= 1)).astype(int)
    t["bike_buffered"] = pd.concat(
        [c == "buffered" for c in classes.values()], axis=1
    ).any(axis=1).astype(int)

    t["bicycle"] = _lower(_col(seg, "osm_bicycle")).fillna("unknown")

    sides = pd.concat([_lower(_col(seg, f)) == "separate" for f, _ in SIDE_FIELDS], axis=1)
    t["parallel_track"] = ((t["highway"] == "cycleway") | sides.any(axis=1)).astype(int)

    parking = pd.concat([_lower(_col(seg, c)) for c in PARKING_COLS], axis=1)
    has_parking = parking.notna() & ~parking.isin(["no", "none"])
    t["parking"] = has_parking.any(axis=1).astype(int)

    # Turn lanes: combine the undirected tag with the forward/backward tags.
    parsed = [
        pd.DataFrame(
            _col(seg, c).map(parse_turn_lanes).tolist(),
            columns=["n", "left", "right"], index=seg.index,
        )
        for c in ("osm_turn_lanes", "osm_turn_lanes_forward", "osm_turn_lanes_backward")
    ]
    t["turn_lanes"] = pd.concat([p["n"] for p in parsed], axis=1).max(axis=1)
    t["turn_left"] = pd.concat([p["left"] for p in parsed], axis=1).max(axis=1).astype(int)
    t["turn_right"] = pd.concat([p["right"] for p in parsed], axis=1).max(axis=1).astype(int)
    return t


def osm_features(segments) -> pd.DataFrame:
    """Sub-block features from matched OSM segments (index dc_subblockkey, osmf_* cols).

    Road segments are aggregated per sub-block (all segments if it has no road).
    osmf_has_parallel_track and osmf_n_segments use every matched segment.
    """
    seg = pd.DataFrame(segments.drop(columns="geometry", errors="ignore"))
    seg = seg[seg["dc_subblockkey"].notna()]
    t = _segment_table(seg)

    has_road = t.groupby("key")["is_road"].transform("any")
    kept = t[t["is_road"] | ~has_road]
    g = kept.groupby("key")

    out = pd.DataFrame(index=pd.Index(sorted(t["key"].unique()), name="dc_subblockkey"))
    out.index = out.index.astype("string")
    hw = _weighted_mode(kept, "key", "highway", "w")
    out["osmf_highway"] = pd.Categorical(hw.reindex(out.index).astype(object))
    out["osmf_speed"] = g["speed"].max()
    out["osmf_lanes"] = g["lanes"].max()
    out["osmf_width"] = g["width"].max()
    out["osmf_oneway"] = g["oneway"].max()
    out["osmf_contraflow"] = g["contraflow"].max()
    out["osmf_bike_best"] = g["bike_best"].max()
    out["osmf_bike_both_sides"] = g["bike_both"].max()
    out["osmf_bike_buffered"] = g["bike_buffered"].max()
    bike = _weighted_mode(kept, "key", "bicycle", "w")
    out["osmf_bicycle_access"] = pd.Categorical(bike.reindex(out.index).astype(object))
    out["osmf_has_parallel_track"] = t.groupby("key")["parallel_track"].max()
    out["osmf_parking_any"] = g["parking"].max()
    out["osmf_turn_lanes"] = g["turn_lanes"].max()
    out["osmf_turn_left"] = g["turn_left"].max()
    out["osmf_turn_right"] = g["turn_right"].max()
    out["osmf_n_segments"] = t.groupby("key").size()
    return out[OUT_COLUMNS]


def fill_rates(feats: pd.DataFrame) -> pd.Series:
    """Share of non-null values per column (binary flags: share equal to 1 in print_report)."""
    return feats.notna().mean()


def print_report(feats: pd.DataFrame) -> None:
    """Print row count, non-null fill rate and positive share for 0/1 flags."""
    print(f"rows={len(feats):,} unique_index={feats.index.is_unique}")
    for c in feats.columns:
        nn = feats[c].notna().mean()
        extra = ""
        if feats[c].dropna().isin([0, 1]).all() and c != "osmf_n_segments":
            extra = f"  share_1={(feats[c] == 1).mean():.3f}"
        print(f"{c:26s} non-null={nn:.3f}{extra}")


if __name__ == "__main__":
    from src.config import load_config
    from src.data.load_snapshot import load_snapshot

    feats = osm_features(load_snapshot(load_config()))
    print_report(feats)
