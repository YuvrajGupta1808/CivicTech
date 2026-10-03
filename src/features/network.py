"""Network-topology features per DC sub-block, from the osmnx-derived snapshot segments.

The graph is undirected and built from ALL snapshot segments (matched or not) whose
``osm_highway`` is a motor road (not cycleway/path/footway/...). Node degree = number of
DISTINCT neighbour nodes, so a two-way street stored as both u->v and v->u counts once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = "dc_subblockkey"
NON_ROAD = {"cycleway", "path", "footway", "pedestrian", "steps", "bridleway", "track"}
ARTERIAL = {"primary", "secondary", "trunk"}  # plus every *_link


def _is_arterial(hw: pd.Series) -> pd.Series:
    s = hw.astype("string")
    return (s.isin(list(ARTERIAL)) | s.str.endswith("_link")).fillna(False).astype(bool)


def _degree(u: np.ndarray, v: np.ndarray) -> pd.Series:
    """Distinct-neighbour degree on the undirected graph (self loops ignored)."""
    keep = u != v
    a = np.minimum(u[keep], v[keep])
    b = np.maximum(u[keep], v[keep])
    e = pd.DataFrame({"a": a, "b": b}).drop_duplicates()
    nodes = np.concatenate([e["a"].to_numpy(), e["b"].to_numpy()])
    return pd.Series(nodes).value_counts()


def network_features(segments) -> pd.DataFrame:
    """``segments``: GeoDataFrame/DataFrame of the snapshot segments with osm_u, osm_v, osm_highway,
    seg_length_m, dc_subblockkey (NaN for unmatched), dc_LENGTH. Returns net_* indexed by dc_subblockkey."""
    cols = ["osm_u", "osm_v", "osm_highway", "seg_length_m", KEY]
    if "dc_LENGTH" in segments.columns:
        cols.append("dc_LENGTH")
    seg = pd.DataFrame(segments[cols]).reset_index(drop=True)
    seg["is_road"] = ~seg["osm_highway"].astype("string").isin(list(NON_ROAD)).fillna(False).to_numpy(dtype=bool)
    seg["is_art"] = _is_arterial(seg["osm_highway"]) & seg["is_road"]

    u = seg["osm_u"].to_numpy()
    v = seg["osm_v"].to_numpy()
    road = seg["is_road"].to_numpy()
    deg_road = _degree(u[road], v[road])
    deg_all = _degree(u, v)

    # arterial segments touching each node (segment-level count over the whole road network)
    art = seg["is_art"].to_numpy()
    art_nodes = pd.Series(np.concatenate([u[art & (u != v)], v[art & (u != v)], u[art & (u == v)]])).value_counts()

    m = seg[seg[KEY].notna()].copy()
    # per sub-block: road segments only, unless it has none
    has_road = m.groupby(KEY)["is_road"].transform("any")
    m = m[m["is_road"] | ~has_road].copy()
    m["fallback"] = ~has_road.loc[m.index]

    g = m.groupby(KEY, sort=True)
    out = pd.DataFrame(index=pd.Index(sorted(g.groups.keys()), name=KEY))
    out.index = out.index.astype("string")

    seg_len = g["seg_length_m"].sum()
    out["net_n_segments"] = g.size().astype("float64")
    if "dc_LENGTH" in m.columns:
        dcl = pd.to_numeric(g["dc_LENGTH"].first(), errors="coerce")
        length = dcl.where(dcl > 0, seg_len)
    else:
        length = seg_len
    length = length.fillna(seg_len)
    out["net_length_m"] = length.astype("float64")
    out["net_log_length"] = np.log1p(out["net_length_m"])

    # long table: (sub-block, node) incl. fallback sub-blocks (use full-graph degree there)
    ends = pd.concat([
        m[[KEY, "fallback", "is_art"]].assign(node=m["osm_u"].to_numpy()),
        m.loc[m["osm_u"] != m["osm_v"], [KEY, "fallback", "is_art"]].assign(
            node=m.loc[m["osm_u"] != m["osm_v"], "osm_v"].to_numpy()),
    ], ignore_index=True)  # self loops contribute their node once (matches art_nodes)
    d_road = ends["node"].map(deg_road)
    d_all = ends["node"].map(deg_all)
    ends["deg"] = np.where(ends["fallback"], d_all, d_road)
    ends["deg"] = ends["deg"].fillna(0).astype(float)

    nodes = ends.drop_duplicates([KEY, "node"])
    gn = nodes.groupby(KEY, sort=True)
    out["net_max_legs"] = gn["deg"].max().astype("float64")
    n_j = nodes.assign(j=(nodes["deg"] >= 3)).groupby(KEY)["j"].sum()
    out["net_n_junctions"] = n_j.astype("float64")
    out["net_junctions_per_100m"] = out["net_n_junctions"] / out["net_length_m"].where(out["net_length_m"] > 0) * 100.0

    # endpoint arterial: sub-block's own length-weighted class is non-arterial, yet one of its
    # nodes is touched by an arterial segment that is NOT one of its own segments.
    w = m.assign(art_len=np.where(m["is_art"], m["seg_length_m"], 0.0))
    art_len = w.groupby(KEY)["art_len"].sum()
    tot_len = w.groupby(KEY)["seg_length_m"].sum()
    own_arterial = (art_len * 2 > tot_len)            # length-weighted majority class is arterial
    own_cnt = ends[ends["is_art"]].groupby([KEY, "node"]).size().rename("own")
    nd = nodes[[KEY, "node"]].merge(own_cnt.reset_index(), on=[KEY, "node"], how="left")
    nd["own"] = nd["own"].fillna(0)
    # own_cnt counts node incidences of own arterial segments (self loops count 2: harmless)
    nd["touch"] = nd["node"].map(art_nodes).fillna(0) - nd["own"]
    ext = (nd["touch"] > 0).groupby(nd[KEY]).any()
    out["net_endpoint_arterial"] = (ext.reindex(out.index).fillna(False) & ~own_arterial.reindex(out.index).fillna(False)).astype("float64")
    return out[["net_length_m", "net_log_length", "net_max_legs", "net_n_junctions",
                "net_junctions_per_100m", "net_endpoint_arterial", "net_n_segments"]]
