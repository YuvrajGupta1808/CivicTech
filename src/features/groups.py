"""Feature groups (for the E5 drop-one-group ablation) and feature sources (for E6).

Column names must match src/features/ddot.py, osm.py (osmf_*) and network.py (net_*).
``select_features`` only ever returns columns that exist in the table it is given.
"""
from __future__ import annotations

from typing import Iterable

FEATURE_GROUPS: dict[str, list[str]] = {
    "speed": ["ddot_speed_max", "osmf_speed"],
    "lanes_width": [
        "ddot_lanes_total", "ddot_lanes_in", "ddot_lanes_out", "ddot_lanes_bidir", "ddot_oneway",
        "ddot_travel_width", "ddot_cross_section_width", "ddot_width_per_lane",
        "osmf_lanes", "osmf_width", "osmf_oneway",
    ],
    "parking": ["ddot_parking_lanes", "ddot_parking_width", "osmf_parking_any"],
    "bike": [
        "ddot_bike_best", "ddot_bike_contraflow", "ddot_bike_next_to_parking",
        "ddot_bike_next_to_through", "ddot_bike_next_to_pocket", "ddot_bike_lanes",
        "ddot_bike_width", "ddot_raised_buffers", "ddot_raised_buffer_width",
        "osmf_contraflow", "osmf_bike_best", "osmf_bike_both_sides", "osmf_bike_buffered",
        "osmf_bicycle_access", "osmf_has_parallel_track",
    ],
    "traffic": ["ddot_log_aadt", "ddot_log_trucks", "ddot_truck_share"],
    "class": ["ddot_fhwa_class", "ddot_dc_class", "ddot_roadtype", "ddot_nhs", "osmf_highway"],
    "conflicts": [
        "ddot_curb_turn_len", "ddot_bus_lane",
        "osmf_turn_lanes", "osmf_turn_left", "osmf_turn_right",
    ],
    "calming": ["ddot_vertical_deflection", "ddot_slow_street", "ddot_double_yellow"],
    "pavement": ["ddot_pci", "ddot_iri", "ddot_surface"],
    "sidewalk": ["ddot_sidewalk_width_min"],
    "network": [
        "net_length_m", "net_log_length", "net_max_legs", "net_n_junctions",
        "net_junctions_per_100m", "net_endpoint_arterial", "net_n_segments", "osmf_n_segments",
    ],
}

SOURCES: dict[str, str] = {"ddot": "ddot_", "osm": "osmf_", "net": "net_"}


def _as_list(x: str | Iterable[str] | None) -> list[str] | None:
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def select_features(columns, groups=None, drop_groups=None, sources=None) -> list[str]:
    """Columns of ``columns`` that belong to the requested groups / sources.

    groups:      keep only these FEATURE_GROUPS keys (None = all groups).
    drop_groups: remove these groups (E5 drop-one-group ablation).
    sources:     keep only these SOURCES keys (None = all), e.g. ["ddot", "net"] (E6).
    Order follows ``columns``; names not in any selected group are excluded; unknown group or
    source names raise KeyError.
    """
    groups, drop_groups, sources = _as_list(groups), _as_list(drop_groups), _as_list(sources)
    for g in (groups or []) + (drop_groups or []):
        if g not in FEATURE_GROUPS:
            raise KeyError(f"unknown feature group {g!r}; known: {sorted(FEATURE_GROUPS)}")
    for s in sources or []:
        if s not in SOURCES:
            raise KeyError(f"unknown source {s!r}; known: {sorted(SOURCES)}")

    keep_groups = list(FEATURE_GROUPS) if groups is None else groups
    keep_groups = [g for g in keep_groups if g not in (drop_groups or [])]
    allowed = {c for g in keep_groups for c in FEATURE_GROUPS[g]}
    prefixes = tuple(SOURCES.values()) if sources is None else tuple(SOURCES[s] for s in sources)
    seen: set[str] = set()
    out: list[str] = []
    for c in columns:
        if c in allowed and c.startswith(prefixes) and c not in seen:
            seen.add(c)
            out.append(c)
    return out
