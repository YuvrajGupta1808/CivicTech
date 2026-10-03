"""Read the RideScore DC basemap snapshot (GeoParquet, EPSG:4326)."""
import geopandas as gpd

KEY_COLS = ["dc_subblockkey", "dc_blockkey"]


def norm_key(s):
    """Normalise DDOT block keys: stripped lower-case strings, NaN stays NaN."""
    return s.astype("string").str.strip().str.lower()


def load_snapshot(cfg) -> gpd.GeoDataFrame:
    path = cfg["paths"]["snapshot"]
    if not path.exists():
        raise FileNotFoundError(f"snapshot missing: {path}")
    gdf = gpd.read_parquet(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    for col in KEY_COLS:
        gdf[col] = norm_key(gdf[col])
    gdf["seg_length_m"] = gdf.geometry.to_crs(cfg["crs_metric"]).length
    if gdf.duplicated(["osm_u", "osm_v", "osm_key"]).any():
        raise ValueError("segment id (osm_u, osm_v, osm_key) is not unique")
    return gdf
