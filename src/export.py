"""Hand-in export: one row per snapshot segment with the out-of-fold crash-risk level and score."""
from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ATTRIBUTION = ("Contains data © OpenStreetMap contributors (ODbL) and District of Columbia DDOT/MPD data "
               "(CC BY 4.0). Derived database licensed ODbL.")
EXPECTED_ROWS = 28978
OUT_NAME = "crash_risk_by_segment.parquet"
COLUMNS = ["osm_u", "osm_v", "osm_key", "snapshot_date", "dc_subblockkey", "crash_risk_level",
           "crash_risk_score", "crash_count_5yr", "scored"]


def _norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def export_segments(snap, oof_main: pd.DataFrame, table: pd.DataFrame, cfg: dict,
                    expected_rows: int | None = EXPECTED_ROWS) -> pd.DataFrame:
    """Write output/crash_risk_by_segment.parquet and return the frame.

    snap:     snapshot GeoDataFrame (osm_u, osm_v, osm_key, match_status, dc_subblockkey, ...)
    oof_main: out-of-fold predictions of the main model only (dc_subblockkey, model, pred_thresholds, mu)
    table:    sub-block table (dc_subblockkey, crash_count)
    Every score is out-of-fold: the sub-block's own ward was held out when it was predicted.
    """
    if oof_main["model"].nunique() != 1:
        raise ValueError("oof_main must hold exactly one model")
    model = str(oof_main["model"].iloc[0])

    pred = oof_main[["dc_subblockkey", "pred_thresholds", "mu"]].copy()
    pred["dc_subblockkey"] = _norm(pred["dc_subblockkey"])
    if pred["dc_subblockkey"].duplicated().any():
        raise ValueError("oof_main has more than one row per sub-block")
    counts = table[["dc_subblockkey", "crash_count"]].copy()
    counts["dc_subblockkey"] = _norm(counts["dc_subblockkey"])
    if counts["dc_subblockkey"].duplicated().any():
        raise ValueError("table has more than one row per sub-block")

    df = pd.DataFrame({
        "osm_u": snap["osm_u"].to_numpy(),
        "osm_v": snap["osm_v"].to_numpy(),
        "osm_key": snap["osm_key"].to_numpy(),
        "snapshot_date": str(cfg["snapshot_date"]),
        "dc_subblockkey": _norm(snap["dc_subblockkey"]).to_numpy(),
        "match_status": snap["match_status"].to_numpy(),
    })
    df["dc_subblockkey"] = df["dc_subblockkey"].astype("string")
    df = df.merge(pred, on="dc_subblockkey", how="left").merge(counts, on="dc_subblockkey", how="left")

    df["scored"] = (df["match_status"] != "none")
    has_pred = df["pred_thresholds"].notna() & df["mu"].notna()
    if (df["scored"] & ~has_pred).any():
        raise ValueError(f"{int((df['scored'] & ~has_pred).sum())} scored segments lack an out-of-fold prediction")
    if (~df["scored"] & has_pred).any():
        raise ValueError("an unscored segment received a prediction")
    df["crash_risk_level"] = df["pred_thresholds"].astype("Int8")
    df["crash_risk_score"] = df["mu"].astype(float).clip(0.0, 2.0)
    df["crash_count_5yr"] = df["crash_count"].astype("Int32")
    out = df[COLUMNS].copy()
    out["dc_subblockkey"] = out["dc_subblockkey"].astype("string")

    if expected_rows is not None and len(out) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, got {len(out)}")
    if len(out) != len(snap):
        raise AssertionError("row count differs from the snapshot")
    if out.duplicated(["osm_u", "osm_v", "osm_key"]).any():
        raise AssertionError("(osm_u, osm_v, osm_key) is not unique")
    if not out.loc[out["scored"], "crash_risk_level"].isin([0, 1, 2]).all():
        raise AssertionError("levels outside {0, 1, 2}")
    sc = out.loc[out["scored"], "crash_risk_score"]
    if not ((sc >= 0) & (sc <= 2)).all():
        raise AssertionError("scores outside [0, 2]")

    window = cfg["crashes"]
    meta = {
        "attribution": ATTRIBUTION,
        "license": "ODbL-1.0 (derived database); DDOT and MPD inputs CC BY 4.0",
        "model": f"{model} (leave-one-ward-out out-of-fold prediction; thresholds fitted on validation wards)",
        "crash_risk_level": "0 = no crash expected, 1 = one crash, 2 = two or more bike crashes in the label window",
        "crash_risk_score": "expected crash level in [0, 2] (p1 + 2*p2); where crashes happened, not risk per ride",
        "label_window": f"{window['window_start']} to {cfg['snapshot_date']} (bike-involved crashes, Crashes in DC)",
        "snapshot_date": str(cfg["snapshot_date"]),
        "unit": "DDOT sub-block; segments of the same sub-block share one level and score",
        "unscored": "segments with match_status == none have no DDOT sub-block (scored = False)",
    }
    out_path = cfg["paths"]["output_dir"] / OUT_NAME
    tbl = pa.Table.from_pandas(out, preserve_index=False)
    merged = dict(tbl.schema.metadata or {})
    merged.update({k.encode(): v.encode() for k, v in meta.items()})
    tbl = tbl.replace_schema_metadata(merged)
    pq.write_table(tbl, out_path, compression="zstd")

    check = pq.read_schema(out_path).metadata
    assert check[b"attribution"].decode() == ATTRIBUTION
    json.loads(check[b"pandas"].decode())  # pandas metadata survived
    return out
