"""Synthetic checks for BNA segment stress (real bikescore-bna rule table)."""
import numpy as np
import pandas as pd

from src.models.bna import bna_stress


def _row(key, hw, speed, lanes, bike=0.0, oneway=0.0, bike_width=0.0, bike_lanes=0.0, fhwa=np.nan):
    return {"dc_subblockkey": key, "osmf_highway": hw, "ddot_speed_max": speed, "ddot_lanes_total": lanes,
            "ddot_bike_best": bike, "ddot_oneway": oneway, "ddot_bike_width": bike_width,
            "ddot_bike_lanes": bike_lanes, "ddot_fhwa_class": fhwa}


def _stress(*rows):
    return bna_stress(pd.DataFrame(rows))


def test_protected_lane_on_arterial_is_comfortable():
    s = _stress(_row("a", "secondary", 25, 2, bike=3, bike_width=10, bike_lanes=2))
    assert s["a"] == 1


def test_fast_wide_road_without_facility_is_uncomfortable():
    s = _stress(_row("b", "primary", 40, 4))
    assert s["b"] == 3


def test_quiet_residential_street_is_comfortable():
    s = _stress(_row("c", "residential", 25, 2))
    assert s["c"] == 1


def test_missing_inputs_use_defaults_and_index_is_subblockkey():
    s = _stress(_row("d", "residential", np.nan, np.nan), _row("e", None, np.nan, np.nan, fhwa="3"))
    assert list(s.index) == ["d", "e"] and set(s.unique()) <= {1, 3}
    assert s["d"] == 1  # residential, BNA default 25 mph / 1 lane
    assert s["e"] == 3  # no OSM class: DDOT FHWA 3 -> primary, BNA default 40 mph


def test_multilane_residential_is_promoted_to_tertiary():
    s = _stress(_row("f", "residential", 25, 4))  # 2 lanes per direction -> tertiary -> stress 3
    assert s["f"] == 3
