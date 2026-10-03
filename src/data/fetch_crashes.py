"""Fetch bike-involved crashes from Crashes in DC (ArcGIS REST) and cache to Parquet."""
import time

import pandas as pd
import requests

from src.data.load_snapshot import norm_key


def _where(cfg) -> str:
    c = cfg["crashes"]
    return (f"{c['primary_filter']} AND REPORTDATE >= DATE '{c['window_start']}' "
            f"AND REPORTDATE < DATE '{c['window_end']}'")


def _get(url, params, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            js = r.json()
            if "error" in js:
                raise RuntimeError(js["error"])
            return js
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def _download(cfg) -> pd.DataFrame:
    c = cfg["crashes"]
    base = {"where": _where(cfg), "outFields": ",".join(c["fields"]), "returnGeometry": "false",
            "orderByFields": "OBJECTID", "f": "json"}
    expected = _get(c["api"], {**base, "returnCountOnly": "true"})["count"]
    rows, offset = [], 0
    while True:
        js = _get(c["api"], {**base, "resultOffset": offset, "resultRecordCount": c["page_size"]})
        feats = js.get("features", [])
        rows += [f["attributes"] for f in feats]
        if not feats or not js.get("exceededTransferLimit", False) and len(feats) < c["page_size"]:
            break
        offset += len(feats)
    df = pd.DataFrame(rows).drop_duplicates("OBJECTID")
    if len(df) != expected:
        raise RuntimeError(f"fetched {len(df)} crashes, server count {expected}")
    df["REPORTDATE"] = pd.to_datetime(df["REPORTDATE"], unit="ms", utc=True)
    return df


def fetch_crashes(cfg, refresh: bool = False) -> pd.DataFrame:
    """Bike crashes in the label window with normalised keys and derived flags.

    Adds: subblockkey, blockkey (normalised, NaN for the 'Route not found' sentinel),
    located, injured (cyclist injured or killed), fatal, near_int (OFFINTERSECTION <= cutoff m).
    """
    cache = cfg["paths"]["crashes_cache"]
    if refresh or not cache.exists():
        _download(cfg).to_parquet(cache, index=False)
    df = pd.read_parquet(cache)
    sentinel = cfg["crashes"]["sentinel"].lower()
    for raw, out in [("SUBBLOCKKEY", "subblockkey"), ("BLOCKKEY", "blockkey")]:
        key = norm_key(df[raw])
        df[out] = key.mask(key == sentinel)
    df["located"] = df["subblockkey"].notna()
    inj = ["MAJORINJURIES_BICYCLIST", "MINORINJURIES_BICYCLIST", "FATAL_BICYCLIST"]
    df["injured"] = df[inj].fillna(0).sum(axis=1) > 0
    df["fatal"] = df["FATAL_BICYCLIST"].fillna(0) > 0
    cutoff = cfg["labels"]["intersection_cutoff_m"]
    df["near_int"] = df["OFFINTERSECTION"] <= cutoff
    return df
