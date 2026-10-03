"""DDOT-derived features, one row per DC sub-block (``dc_*`` columns in, ``ddot_*`` out).

Value formats seen in the snapshot (see docs/memory.md):
  * lane / width / parking counts are floats; ``dc_SUMMARYDIRECTION`` is 'BD' | 'OB' | 'IB' | '??'
    ('??' = direction unknown, lanes are all zero there).
  * ``dc_BIKELANE_*`` hold a direction code ('IB'/'OB'/'BD') when the facility exists, else NaN.
  * ``dc_DOUBLEYELLOW_LINE`` ('Yes'), ``dc_VERTICAL_DEFLECTION`` ('YES'), ``dc_BUSLANE_*``
    ('L1'/'L2'), ``dc_SLOWSTREETINFO`` (free text) are present-or-NaN flags.
  * ``dc_SIDEWALK_*_WIDTH`` are strings ('6', '16+', ...).
  * functional classes / surface type are float codes; ``dc_ROADTYPE`` is a string code.
  * ``dc_IB`` speed columns are 100% null; every AADT year is 2020 (no year flag).
Never emits ward / ANC / SMD / quadrant / IDs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = "dc_subblockkey"


# --------------------------------------------------------------------------- helpers
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric float64 column (NaN when missing / unparsable / column absent)."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def _present(df: pd.DataFrame, col: str) -> pd.Series:
    """Boolean: the cell holds a non-empty, non-'no'-like value."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    txt = s.astype("string").str.strip().str.lower()
    absent = s.isna() | txt.isna() | txt.isin(["", "no", "n", "false", "0", "none", "nan"])
    return pd.Series(~absent.to_numpy(dtype=bool), index=df.index)


def _flag(df: pd.DataFrame, *cols: str) -> pd.Series:
    """0/1 float: any of the columns present (absent == 0, these are sparse 'is mapped' flags)."""
    out = pd.Series(False, index=df.index)
    for c in cols:
        out = out | _present(df, c)
    return out.astype("float64")


def _category(s: pd.Series) -> pd.Series:
    """Category dtype. Numeric codes -> integer categories (4.0 -> 4); text -> string labels.

    NB: sklearn 1.8 HistGradientBoosting(categorical_features="from_dtype") fails on pandas-3
    ``string`` categories that contain NA, so labels go through ``object`` and codes stay ints.
    """
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() >= s.notna().sum():
        return num.round().astype("Int64").astype("category")
    lab = s.astype("string").str.strip()
    return lab.mask(lab == "").astype(object).where(lab.notna(), np.nan).astype("category")


def _width_text(df: pd.DataFrame, col: str) -> pd.Series:
    """'16+' -> 16.0, '6' -> 6.0, NaN stays NaN."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    ext = df[col].astype("string").str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(ext, errors="coerce").astype("float64")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.where(b > 0)).astype("float64")


# --------------------------------------------------------------------------- main
def ddot_features(sub: pd.DataFrame) -> pd.DataFrame:
    """Build ``ddot_*`` features. ``sub``: one row per sub-block with ``dc_*`` columns
    (``dc_subblockkey`` as a column or as the index). Returns a frame indexed by dc_subblockkey."""
    df = sub
    if KEY not in df.columns:
        if df.index.name == KEY:
            df = df.reset_index()
        else:
            raise KeyError(f"{KEY} must be a column or the index name")
    df = df.drop_duplicates(KEY).reset_index(drop=True)

    out = pd.DataFrame(index=pd.Index(df[KEY].astype("string"), name=KEY))
    o: dict[str, pd.Series] = {}

    # ---- lanes
    lanes = _num(df, "dc_TOTALTRAVELLANES")
    lin = _num(df, "dc_TOTALTRAVELLANESINBOUND")
    lout = _num(df, "dc_TOTALTRAVELLANESOUTBOUND")
    lbi = _num(df, "dc_TOTALTRAVELLANESBIDIRECTIONAL")
    o["ddot_lanes_total"] = lanes
    o["ddot_lanes_in"] = lin
    o["ddot_lanes_out"] = lout
    o["ddot_lanes_bidir"] = lbi
    # one-way from SUMMARYDIRECTION (OB/IB = one-way, BD = two-way, '??' unknown), else from in/out lanes
    if "dc_SUMMARYDIRECTION" in df.columns:
        d = df["dc_SUMMARYDIRECTION"].astype("string").str.strip().str.upper()
    else:
        d = pd.Series(pd.NA, index=df.index, dtype="string")
    ow = pd.Series(np.nan, index=df.index, dtype="float64")
    ow[d.isin(["OB", "IB"]).fillna(False).to_numpy(dtype=bool)] = 1.0
    ow[(d == "BD").fillna(False).to_numpy(dtype=bool)] = 0.0
    unk = ow.isna()
    ow[unk & ((lin > 0) ^ (lout > 0)) & (lbi.fillna(0) == 0)] = 1.0
    ow[ow.isna() & (lin > 0) & (lout > 0)] = 0.0
    ow[ow.isna() & (lbi > 0)] = 0.0
    o["ddot_oneway"] = ow

    # ---- widths (feet in the source)
    tw = _num(df, "dc_TOTALTRAVELLANEWIDTH")
    o["ddot_travel_width"] = tw
    o["ddot_cross_section_width"] = _num(df, "dc_TOTALCROSSSECTIONWIDTH")
    o["ddot_width_per_lane"] = _safe_div(tw, lanes)

    # ---- parking
    o["ddot_parking_lanes"] = _num(df, "dc_TOTALPARKINGLANES")
    o["ddot_parking_width"] = _num(df, "dc_TOTALPARKINGLANEWIDTH")

    # ---- speed: OB, fall back to OB_ALT; 0 is a placeholder, not a limit
    sp = _num(df, "dc_SPEEDLIMITS_OB").where(lambda s: s > 0)
    alt = _num(df, "dc_SPEEDLIMITS_OB_ALT").where(lambda s: s > 0)
    o["ddot_speed_max"] = sp.fillna(alt)

    # ---- classes
    o["ddot_fhwa_class"] = _category(df["dc_FHWAFUNCTIONALCLASS"]) if "dc_FHWAFUNCTIONALCLASS" in df else _empty_cat(df)
    o["ddot_dc_class"] = _category(df["dc_DCFUNCTIONALCLASS"]) if "dc_DCFUNCTIONALCLASS" in df else _empty_cat(df)
    o["ddot_roadtype"] = _category(df["dc_ROADTYPE"]) if "dc_ROADTYPE" in df else _empty_cat(df)
    o["ddot_nhs"] = _flag(df, "dc_NHSTYPE", "dc_NHSCODE")

    # ---- bike facilities (direction codes; any non-empty value = facility present)
    prot = _flag(df, "dc_BIKELANE_PROTECTED", "dc_BIKELANE_DUAL_PROTECTED")
    buff = _flag(df, "dc_BIKELANE_BUFFERED", "dc_BIKELANE_DUAL_BUFFERED")
    conv = _flag(df, "dc_BIKELANE_CONVENTIONAL")
    nbike = _num(df, "dc_TOTALBIKELANES")
    rank = np.where(prot > 0, 3.0, np.where(buff > 0, 2.0, np.where((conv > 0) | (nbike > 0), 1.0, 0.0)))
    o["ddot_bike_best"] = pd.Series(rank, index=df.index, dtype="float64")
    o["ddot_bike_contraflow"] = _flag(df, "dc_BIKELANE_CONTRAFLOW")
    o["ddot_bike_next_to_parking"] = _flag(df, "dc_BIKELANE_PARKINGLANE_ADJACENT")
    o["ddot_bike_next_to_through"] = _flag(df, "dc_BIKELANE_THROUGHLANE_ADJACENT")
    o["ddot_bike_next_to_pocket"] = _flag(df, "dc_BIKELANE_POCKETLANE_ADJACENT")
    o["ddot_bike_lanes"] = nbike
    o["ddot_bike_width"] = _num(df, "dc_TOTALBIKELANEWIDTH")
    o["ddot_raised_buffers"] = _num(df, "dc_TOTALRAISEDBUFFERS")
    o["ddot_raised_buffer_width"] = _num(df, "dc_TOTALRAISEDBUFFERWIDTH")

    # ---- traffic (AADT only exists for ~1/3 of sub-blocks; NaN elsewhere, GBM handles it)
    aadt = _num(df, "dc_AADT")
    su, co = _num(df, "dc_AADT_SINGLE_UNIT"), _num(df, "dc_AADT_COMBINATION")
    trucks = pd.concat([su, co], axis=1).sum(axis=1, min_count=1)
    o["ddot_log_aadt"] = np.log1p(aadt.clip(lower=0))
    o["ddot_log_trucks"] = np.log1p(trucks.clip(lower=0))
    o["ddot_truck_share"] = _safe_div(trucks, aadt).clip(upper=1.0)

    # ---- conflicts
    curb = pd.concat([_num(df, "dc_LEFTTURN_CURBLANE_EXCL_LEN"), _num(df, "dc_RIGHTTURN_CURBLANE_EXCL_LEN")],
                     axis=1).sum(axis=1, min_count=1)
    o["ddot_curb_turn_len"] = curb
    o["ddot_bus_lane"] = _flag(df, "dc_BUSLANE_INBOUND", "dc_BUSLANE_OUTBOUND")

    # ---- calming / markings
    o["ddot_vertical_deflection"] = _flag(df, "dc_VERTICAL_DEFLECTION")
    o["ddot_slow_street"] = _flag(df, "dc_SLOWSTREETINFO")
    o["ddot_double_yellow"] = _flag(df, "dc_DOUBLEYELLOW_LINE")

    # ---- pavement
    o["ddot_pci"] = _num(df, "dc_PCI_SCORE")
    o["ddot_iri"] = _num(df, "dc_IRI")
    o["ddot_surface"] = _category(df["dc_SURFACE_TYPE"]) if "dc_SURFACE_TYPE" in df else _empty_cat(df)

    # ---- sidewalk (min of the two sides that exist)
    sw = pd.concat([_width_text(df, "dc_SIDEWALK_IB_WIDTH"), _width_text(df, "dc_SIDEWALK_OB_WIDTH")], axis=1)
    o["ddot_sidewalk_width_min"] = sw.min(axis=1, skipna=True)

    res = pd.DataFrame({k: v.to_numpy() if not isinstance(v.dtype, pd.CategoricalDtype) else v.array
                        for k, v in o.items()}, index=out.index)
    return res


def _empty_cat(df: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.Categorical([None] * len(df)), index=df.index)
