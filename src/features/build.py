"""Join labels and feature groups into one modelling row per DDOT sub-block."""
import numpy as np
import pandas as pd

from src.data.fetch_crashes import fetch_crashes
from src.data.load_snapshot import load_snapshot
from src.features.ddot import ddot_features
from src.features.network import network_features
from src.labels import build_block_labels, build_labels, join_report

FEATURE_PREFIXES = ("ddot_", "osmf_", "net_")
CONSTANT_CHECK = ["dc_TOTALTRAVELLANES", "dc_SPEEDLIMITS_OB", "dc_AADT", "dc_WARD_ID", "dc_blockkey"]


def _matched(snap):
    keep = snap["match_status"].ne("none") & snap["dc_subblockkey"].notna()
    return snap[keep]


def _assert_constant(matched):
    for col in CONSTANT_CHECK:
        n = matched.groupby("dc_subblockkey")[col].nunique(dropna=True)
        if (n > 1).any():
            raise ValueError(f"{col} varies within {(n > 1).sum()} sub-blocks")


def _osm(matched):
    try:
        from src.features.osm import osm_features
    except ImportError as exc:  # cut line: OSM features not ready
        print(f"[build] OSM features skipped: {exc}")
        return None
    return osm_features(matched)


def _raw_lts(sub):
    """Production RideScore LTS v1 from raw DDOT fields (reference, never a feature)."""
    try:
        from src.models.baselines import LTS_RAW_FIELDS, lts_from_raw
    except ImportError as exc:
        print(f"[build] LTS skipped: {exc}")
        return None
    raw = pd.DataFrame(index=sub.index)
    for field in LTS_RAW_FIELDS:
        src = "dc_blockkey" if field == "BLOCKKEY" else f"dc_{field}"
        raw[f"raw_{field}"] = sub[src] if src in sub else np.nan
    return pd.DataFrame({"ref_lts": lts_from_raw(raw)}, index=sub.index)


def _drop_uninformative(table, max_null=0.99):
    feats = [c for c in table if c.startswith(FEATURE_PREFIXES)]
    drop = [c for c in feats
            if table[c].isna().mean() > max_null or table[c].nunique(dropna=True) <= 1]
    return table.drop(columns=drop), drop


def build_table(cfg):
    """Return (table, report). One row per matched sub-block with ward, labels, features."""
    snap = load_snapshot(cfg)
    crashes = fetch_crashes(cfg)
    matched = _matched(snap)
    _assert_constant(matched)

    sub = pd.DataFrame(matched.drop(columns="geometry")).drop_duplicates("dc_subblockkey")
    sub = sub.set_index("dc_subblockkey", drop=False)
    sub.index.name = None

    base = pd.DataFrame(index=sub.index)
    base["dc_subblockkey"] = sub["dc_subblockkey"]
    base["dc_blockkey"] = sub["dc_blockkey"]
    base["ward"] = pd.to_numeric(sub["dc_WARD_ID"], errors="coerce")
    weak = matched["match_status"].eq("weak").groupby(matched["dc_subblockkey"]).all()
    base["weak_match"] = weak.reindex(base.index).fillna(False).astype(bool)

    parts = [ddot_features(sub.reset_index(drop=True)), network_features(snap)]
    osm = _osm(matched)
    if osm is not None:
        parts.append(osm)
    lts = _raw_lts(sub)
    if lts is not None:
        parts.append(lts)
    table = base.join(parts, how="left")

    labels = build_labels(sub.index, crashes, cfg).set_index("dc_subblockkey")
    table = table.join(labels, how="left")
    blocks = build_block_labels(sub["dc_blockkey"].dropna().unique(), crashes)
    table = table.merge(blocks, on="dc_blockkey", how="left").set_index("dc_subblockkey", drop=False)
    table.index.name = None
    table["length_m"] = table["net_length_m"]

    n_no_ward = int(table["ward"].isna().sum())
    table = table[table["ward"].notna()].copy()
    table["ward"] = table["ward"].astype(int)
    table, dropped = _drop_uninformative(table)

    report = join_report(sub.index, sub["dc_blockkey"].dropna().unique(), crashes)
    report.update({"n_subblocks": int(len(table)), "n_dropped_no_ward": n_no_ward,
                   "dropped_feature_cols": dropped,
                   "n_features": int(sum(c.startswith(FEATURE_PREFIXES) for c in table))})
    return table.reset_index(drop=True), report


def feature_columns(table):
    return [c for c in table.columns if c.startswith(FEATURE_PREFIXES)]
