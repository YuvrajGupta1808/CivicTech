"""Unit tests for OSM tag parsers and the per-sub-block aggregation."""
import numpy as np
import pandas as pd
import pytest

from src.features.osm import (
    FACILITY_RANK,
    OUT_COLUMNS,
    normalise_cycleway,
    osm_features,
    parse_lanes,
    parse_maxspeed,
    parse_turn_lanes,
    parse_width,
)


@pytest.mark.parametrize("text, expected", [
    ("25 mph", 25.0),
    ("25", 25.0),
    ("24.14 mph", 24.14),
    ("40 km/h", 40 / 1.609344),
    ("50 kph", 50 / 1.609344),
    ("25 mph;30 mph", 25.0),
])
def test_parse_maxspeed_values(text, expected):
    assert parse_maxspeed(text) == pytest.approx(expected)


def test_parse_maxspeed_km_h_is_about_24_85():
    assert parse_maxspeed("40 km/h") == pytest.approx(24.85, abs=0.01)


@pytest.mark.parametrize("text", ["none", "signals", "walk", "US:urban", "", None, np.nan, pd.NA])
def test_parse_maxspeed_missing_or_junk(text):
    assert np.isnan(parse_maxspeed(text))


@pytest.mark.parametrize("text, expected", [("2", 2.0), ("4", 4.0), ("2;3", 3.0), ("3;2", 3.0)])
def test_parse_lanes(text, expected):
    assert parse_lanes(text) == expected


@pytest.mark.parametrize("text", [None, np.nan, pd.NA, "", "many"])
def test_parse_lanes_missing(text):
    assert np.isnan(parse_lanes(text))


def test_parse_turn_lanes_pipe_counts():
    assert parse_turn_lanes("left|through|right") == (3.0, 1, 1)
    assert parse_turn_lanes("left||") == (3.0, 1, 0)
    assert parse_turn_lanes("||right") == (3.0, 0, 1)
    assert parse_turn_lanes("through") == (1.0, 0, 0)
    assert parse_turn_lanes("left;through|through;right") == (2.0, 1, 1)


@pytest.mark.parametrize("text", [None, np.nan, pd.NA, ""])
def test_parse_turn_lanes_missing(text):
    n, left, right = parse_turn_lanes(text)
    assert np.isnan(n) and left == 0 and right == 0


def test_parse_width_units():
    assert parse_width("3.5") == pytest.approx(3.5)
    assert parse_width("31'0\"") == pytest.approx(31 * 0.3048)
    assert parse_width("20'8\"") == pytest.approx(20 * 0.3048 + 8 * 0.0254)
    assert parse_width("12 ft") == pytest.approx(12 * 0.3048)
    assert np.isnan(parse_width(None))
    assert np.isnan(parse_width("wide"))


@pytest.mark.parametrize("value, expected", [
    ("track", "track"),
    ("separate", "track"),
    ("opposite_track", "track"),
    ("lane", "lane"),
    ("opposite_lane", "lane"),
    ("exclusive", "lane"),
    ("shared_lane", "shared"),
    ("share_busway", "shared"),
    ("shoulder", "shared"),
    ("no", "none"),
    ("none", "none"),
    ("crossing", "none"),
    ("traffic_island", "none"),
    ("", "none"),
    (None, "none"),
    (np.nan, "none"),
    (pd.NA, "none"),
    ("lane;shared_lane", "lane"),
])
def test_normalise_cycleway(value, expected):
    assert normalise_cycleway(value) == expected


def test_normalise_cycleway_buffer():
    assert normalise_cycleway("lane", "yes") == "buffered"
    assert normalise_cycleway("opposite_lane", "0.5") == "buffered"
    assert normalise_cycleway("lane", "no") == "lane"
    assert normalise_cycleway("lane", np.nan) == "lane"
    assert normalise_cycleway("track", "yes") == "track"
    assert normalise_cycleway("shared_lane", "yes") == "shared"
    assert normalise_cycleway("no", "yes") == "none"


def test_facility_rank_order():
    assert FACILITY_RANK == {"track": 3, "buffered": 2, "lane": 1, "shared": 0.5, "none": 0}


def _synthetic():
    """Sub-block A: two road segments + one separately mapped cycleway. B: path only."""
    cols = [
        "dc_subblockkey", "seg_length_m", "osm_highway", "osm_maxspeed", "osm_lanes",
        "osm_oneway", "osm_oneway_bicycle", "osm_bicycle", "osm_cycleway_left",
        "osm_cycleway_left_buffer", "osm_cycleway_right", "osm_parking_both",
        "osm_turn_lanes", "osm_width",
    ]
    rows = [
        ["a", 100.0, "residential", "25 mph", "2", True, "no", np.nan,
         "lane", "yes", "lane", "yes", "left|through", np.nan],
        ["a", 20.0, "primary", "40 km/h", "2;3", False, np.nan, np.nan,
         np.nan, np.nan, np.nan, "no", "through|right", "9"],
        ["a", 50.0, "cycleway", np.nan, np.nan, False, np.nan, "designated",
         np.nan, np.nan, np.nan, np.nan, np.nan, "2"],
        ["b", 30.0, "path", np.nan, np.nan, False, np.nan, "yes",
         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [None, 10.0, "residential", "99 mph", "9", False, np.nan, np.nan,
         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]
    return pd.DataFrame(rows, columns=cols)


def test_osm_features_three_segment_block():
    feats = osm_features(_synthetic())
    assert list(feats.columns) == OUT_COLUMNS
    assert list(feats.index) == ["a", "b"]  # unmatched row dropped
    a = feats.loc["a"]
    # Road segments only: residential (100 m) beats primary (20 m).
    assert a["osmf_highway"] == "residential"
    assert a["osmf_speed"] == 25.0  # max(25 mph, 40 km/h = 24.85 mph)
    assert a["osmf_lanes"] == 3
    assert a["osmf_width"] == 9  # the cycleway's width of 2 is ignored
    assert a["osmf_oneway"] == 1
    assert a["osmf_contraflow"] == 1
    assert a["osmf_bike_best"] == FACILITY_RANK["buffered"]
    assert a["osmf_bike_both_sides"] == 1
    assert a["osmf_bike_buffered"] == 1
    assert a["osmf_bicycle_access"] == "unknown"  # road segments only
    assert a["osmf_has_parallel_track"] == 1  # from the cycleway segment
    assert a["osmf_parking_any"] == 1
    assert a["osmf_turn_lanes"] == 2
    assert a["osmf_turn_left"] == 1 and a["osmf_turn_right"] == 1
    assert a["osmf_n_segments"] == 3


def test_osm_features_non_road_only_block_falls_back():
    b = osm_features(_synthetic()).loc["b"]
    assert b["osmf_highway"] == "path"
    assert b["osmf_bicycle_access"] == "yes"
    assert b["osmf_has_parallel_track"] == 0
    assert b["osmf_bike_best"] == 0
    assert np.isnan(b["osmf_speed"]) and np.isnan(b["osmf_lanes"])
    assert np.isnan(b["osmf_turn_lanes"])
    assert b["osmf_n_segments"] == 1


def test_osm_features_dtypes():
    feats = osm_features(_synthetic())
    assert isinstance(feats["osmf_highway"].dtype, pd.CategoricalDtype)
    assert isinstance(feats["osmf_bicycle_access"].dtype, pd.CategoricalDtype)
    assert feats["osmf_speed"].dtype == float
