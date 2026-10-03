"""Fast unit tests for src.labels on tiny synthetic crash tables."""
import numpy as np
import pandas as pd

from src.labels import build_block_labels, build_labels, join_report, level_from_count

CFG = {"labels": {"intersection_cutoff_m": 15}}


def make_crashes(rows):
    """rows: (subblockkey, blockkey, off_int, injured, fatal, ward)."""
    df = pd.DataFrame(rows, columns=["subblockkey", "blockkey", "OFFINTERSECTION",
                                     "injured", "fatal", "WARD"])
    df["subblockkey"] = df["subblockkey"].astype("string")
    df["blockkey"] = df["blockkey"].astype("string")
    df["located"] = df["subblockkey"].notna()
    df["near_int"] = df["OFFINTERSECTION"] <= CFG["labels"]["intersection_cutoff_m"]
    return df


def sample():
    return make_crashes([
        ("a1", "a", 3.0, True, False, "Ward 1"),
        ("a1", "a", 40.0, False, False, "Ward 1"),
        ("a1", "a", 15.0, True, True, "Ward 1"),     # exactly at cutoff = intersection
        ("a2", "a", 100.0, False, False, "Ward 1"),
        ("b1", "b", np.nan, True, False, "Ward 2"),  # NaN offset: neither split
        (None, None, 5.0, False, False, "Ward 2"),   # sentinel / unlocated
        ("zz", "a", 20.0, False, False, "Ward 2"),   # key not in snapshot
    ])


def test_level_boundaries():
    assert level_from_count([0, 1, 2, 3, 50]).tolist() == [0, 1, 2, 2, 2]
    assert level_from_count([]).tolist() == []


def test_missing_keys_get_zero_and_order_kept():
    lab = build_labels(["c9", "a1", "a1", None], sample(), CFG)
    assert lab["dc_subblockkey"].tolist() == ["c9", "a1"]
    row = lab.set_index("dc_subblockkey").loc["c9"]
    assert (row.to_numpy() == 0).all()


def test_counts_and_levels():
    lab = build_labels(["a1", "a2", "b1"], sample(), CFG).set_index("dc_subblockkey")
    assert lab.loc["a1", "crash_count"] == 3 and lab.loc["a1", "level"] == 2
    assert lab.loc["a2", "crash_count"] == 1 and lab.loc["a2", "level"] == 1
    assert lab.loc["b1", "crash_count"] == 1 and lab.loc["b1", "level"] == 1


def test_sentinel_and_nan_keys_ignored():
    lab = build_labels(["a1", "a2", "b1"], sample(), CFG)
    assert lab["crash_count"].sum() == 5  # sentinel and out-of-snapshot crashes excluded


def test_midblock_vs_intersection_split():
    lab = build_labels(["a1", "b1"], sample(), CFG).set_index("dc_subblockkey")
    assert lab.loc["a1", "int_count"] == 2       # 3 m and 15 m
    assert lab.loc["a1", "midblock_count"] == 1  # 40 m
    assert lab.loc["a1", "level_midblock"] == 1
    assert lab.loc["b1", "int_count"] == 0 and lab.loc["b1", "midblock_count"] == 0
    assert lab.loc["b1", "level_midblock"] == 0


def test_injury_variant():
    lab = build_labels(["a1", "a2"], sample(), CFG).set_index("dc_subblockkey")
    assert lab.loc["a1", "injury_count"] == 2 and lab.loc["a1", "level_injury"] == 2
    assert lab.loc["a1", "fatal_count"] == 1
    assert lab.loc["a2", "injury_count"] == 0 and lab.loc["a2", "level_injury"] == 0


def test_block_labels():
    lab = build_block_labels(["a", "b", "q"], sample()).set_index("dc_blockkey")
    assert lab.loc["a", "crash_count_block"] == 5 and lab.loc["a", "level_block"] == 2
    assert lab.loc["b", "level_block"] == 1
    assert lab.loc["q", "crash_count_block"] == 0 and lab.loc["q", "level_block"] == 0


def test_join_report():
    rep = join_report(["a1", "a2", "b1", "c9"], ["a", "b"], sample())
    assert rep["n_crashes"] == 7 and rep["n_sentinel"] == 1
    assert rep["n_matched_subblock"] == 5
    assert rep["n_unmatched_located"] == 1 and rep["n_unmatched_distinct_keys"] == 1
    assert rep["n_unmatched_but_block_in_snapshot"] == 1
    assert abs(rep["match_rate_located"] - 5 / 6) < 1e-9
    assert abs(rep["match_rate_all"] - 5 / 7) < 1e-9
    assert rep["match_rate_by_ward"] == {"Ward 1": 1.0, "Ward 2": 0.5}
    assert rep["label_counts"] == {"0": 1, "1": 2, "2": 1}
    assert all(isinstance(v, (int, float)) for v in rep["label_counts"].values())
