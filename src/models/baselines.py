"""M0 (majority class) and M1 (RideScore production LTS v1 mapped to 3 levels)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ridescore.models.ridescore_v1.lts import lts_level
from ridescore.network.normalise import classify_facility, name_function, num_lanes, speed_limit

# Raw DDOT fields (no dc_ prefix) the LTS needs; the modelling table carries them as raw_<FIELD>.
LTS_RAW_FIELDS = [
    "BIKELANE_PROTECTED",
    "BIKELANE_DUAL_PROTECTED",
    "BIKELANE_BUFFERED",
    "BIKELANE_CONVENTIONAL",
    "SPEEDLIMITS_OB",
    "TOTALTRAVELLANES",
    "DCFUNCTIONALCLASS",
]

# LTS 1-2 -> level 0, LTS 3 -> level 1, LTS 4 -> level 2
LTS_TO_LEVEL = {1: 0, 2: 0, 3: 1, 4: 2}


def m0_proba(n: int, K: int = 3) -> np.ndarray:
    """All probability mass on the majority class 0."""
    P = np.zeros((n, K))
    P[:, 0] = 1.0
    return P


def lts_from_raw(df: pd.DataFrame) -> pd.Series:
    """RideScore v1 LTS (1..4) per row from raw_<FIELD> columns, using RideScore's own normalisers."""
    missing = [f"raw_{f}" for f in LTS_RAW_FIELDS if f"raw_{f}" not in df.columns]
    if missing:
        raise KeyError(f"lts_from_raw: missing columns {missing}")
    cols = {f: df[f"raw_{f}"].to_numpy(dtype=object) for f in LTS_RAW_FIELDS}
    cache: dict = {}
    out = np.empty(len(df), dtype=np.int64)
    for i in range(len(df)):
        tags = {f: (None if pd.isna(cols[f][i]) else cols[f][i]) for f in LTS_RAW_FIELDS}
        facility = classify_facility(tags)
        speed = speed_limit(tags["SPEEDLIMITS_OB"])
        lanes = num_lanes(tags["TOTALTRAVELLANES"])
        func = name_function(tags["DCFUNCTIONALCLASS"])
        key = (facility, speed, lanes, func)
        if key not in cache:
            cache[key] = int(lts_level(facility, speed, lanes, func))
        out[i] = cache[key]
    return pd.Series(out, index=df.index, name="lts_level")


def m1_lts(lts) -> tuple[np.ndarray, np.ndarray]:
    """LTS (1..4) -> one-hot P (n, 3) and score = mapped level (0/1/2)."""
    lts = np.asarray(lts).astype(int)
    level = np.array([LTS_TO_LEVEL[v] for v in lts], dtype=int)
    P = np.zeros((len(level), 3))
    P[np.arange(len(level)), level] = 1.0
    return P, level.astype(float)
