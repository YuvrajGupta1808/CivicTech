"""Ordinal crash labels per sub-block (and per block) from bike-crash counts.

Level 0 = no crash, 1 = one crash, 2 = two or more, over the label window.
Only located crashes (a real subblockkey, not the 'Route not found' sentinel) count.
"""
import numpy as np
import pandas as pd


def level_from_count(counts) -> np.ndarray:
    """Map crash counts to ordinal levels: 0 -> 0, 1 -> 1, >=2 -> 2."""
    return np.minimum(np.asarray(counts, dtype=int), 2)


def _unique_keys(keys) -> pd.Series:
    """Unique non-null keys as a string Series, first-seen order."""
    s = pd.Series(keys, dtype="string").dropna()
    return s.drop_duplicates().reset_index(drop=True)


def _located(crashes: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Crashes flagged located with a non-null key in key_col."""
    ok = crashes[key_col].notna()
    if "located" in crashes:
        ok &= crashes["located"].fillna(False).astype(bool)
    return crashes[ok]


def _count_by_key(keys: pd.Series, crashes: pd.DataFrame, key_col: str, flags: dict) -> pd.DataFrame:
    """One row per key; each flag column is summed per key (missing keys -> 0)."""
    c = _located(crashes, key_col)
    data = {name: flag.reindex(c.index).fillna(False).astype(int) for name, flag in flags.items()}
    agg = pd.DataFrame(data).groupby(c[key_col].astype("string")).sum()
    out = agg.reindex(pd.Index(keys)).fillna(0).astype(int)
    return out.reset_index(drop=True)


def build_labels(sub_keys, crashes: pd.DataFrame, cfg) -> pd.DataFrame:
    """Sub-block labels: counts (all, injury, fatal, intersection, mid-block) and levels."""
    keys = _unique_keys(sub_keys)
    cutoff = cfg["labels"]["intersection_cutoff_m"]
    off = crashes["OFFINTERSECTION"]
    flags = {
        "crash_count": pd.Series(True, index=crashes.index),
        "injury_count": crashes["injured"],
        "fatal_count": crashes["fatal"],
        "int_count": crashes["near_int"],
        "midblock_count": off > cutoff,  # NaN compares False: neither intersection nor mid-block
    }
    out = _count_by_key(keys, crashes, "subblockkey", flags)
    out.insert(0, "dc_subblockkey", keys)
    out["level"] = level_from_count(out["crash_count"])
    out["level_injury"] = level_from_count(out["injury_count"])
    out["level_midblock"] = level_from_count(out["midblock_count"])
    return out


def build_block_labels(block_keys, crashes: pd.DataFrame) -> pd.DataFrame:
    """Block-level labels via the crash blockkey: dc_blockkey, crash_count_block, level_block."""
    keys = _unique_keys(block_keys)
    flags = {"crash_count_block": pd.Series(True, index=crashes.index)}
    out = _count_by_key(keys, crashes, "blockkey", flags)
    out.insert(0, "dc_blockkey", keys)
    out["level_block"] = level_from_count(out["crash_count_block"])
    return out


def _rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def join_report(sub_keys, block_keys, crashes: pd.DataFrame) -> dict:
    """How well crashes join to the snapshot sub-blocks; JSON-serialisable."""
    sub_set = set(_unique_keys(sub_keys))
    block_set = set(_unique_keys(block_keys))
    loc = _located(crashes, "subblockkey")
    in_sub = loc["subblockkey"].isin(sub_set)
    unmatched = loc[~in_sub]
    n_all, n_loc, n_match = len(crashes), len(loc), int(in_sub.sum())
    ward_rate = {}
    if "WARD" in crashes:
        for ward, g in loc.groupby(loc["WARD"].fillna("unknown").astype(str)):
            ward_rate[str(ward)] = _rate(int(g["subblockkey"].isin(sub_set).sum()), len(g))
    keys = _unique_keys(sub_keys)
    n_per_key = _count_by_key(keys, crashes, "subblockkey", {"n": pd.Series(True, index=crashes.index)})["n"]
    lc = pd.Series(level_from_count(n_per_key)).value_counts()
    return {
        "n_crashes": int(n_all),
        "n_sentinel": int(n_all - n_loc),
        "n_matched_subblock": n_match,
        "match_rate_all": _rate(n_match, n_all),
        "match_rate_located": _rate(n_match, n_loc),
        "n_unmatched_located": int(len(unmatched)),
        "n_unmatched_distinct_keys": int(unmatched["subblockkey"].nunique()),
        "n_unmatched_but_block_in_snapshot": int(unmatched["blockkey"].isin(block_set).sum()),
        "match_rate_by_ward": ward_rate,
        "label_counts": {str(k): int(lc.get(k, 0)) for k in (0, 1, 2)},
    }
