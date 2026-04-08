"""
features.py

Compute engineered features for the stations DataFrame:
  - metro_pop: population of the nearest US city within 50 km
  - distance_to_nearest_major_city_km: Haversine to nearest city with pop > 500k
  - num_nearby_stations: count of other Amtrak TRAIN stations within 80 km
  - is_northeast_corridor: binary flag for NEC bounding box

Expects stations.csv produced by parse_and_join.py.
Writes an updated stations.csv in place (adds columns).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
CITIES_CSV = ROOT / "data" / "us_cities_pop.csv"

# NEC bounding box (approximate): Boston → Washington
# lat: 38.89°N (Union Station DC) to 42.36°N (South Station Boston)
# lon: east of -77.10°W
NEC_LAT_MIN = 38.89
NEC_LAT_MAX = 42.40
NEC_LON_MIN = -77.10  # east of this (i.e., lon ≥ -77.10)

EARTH_RADIUS_KM = 6371.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def deg_to_rad(arr):
    return np.radians(arr)


def build_kdtree_rad(lats, lons):
    """Build a cKDTree using (lat_rad, lon_rad) as 2-D keys.

    Note: cKDTree distances in radian-space approximate arc distances only for
    small separations. We use it as a fast pre-filter and recompute exact
    Haversine for the returned candidates.
    """
    coords = np.column_stack([deg_to_rad(lats), deg_to_rad(lons)])
    return cKDTree(coords)


def haversine_km_vec(lat1, lon1, lat2_arr, lon2_arr):
    """Haversine from one point to an array of points (km)."""
    R = EARTH_RADIUS_KM
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r = np.radians(lat2_arr)
    lon2_r = np.radians(lon2_arr)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Feature functions ──────────────────────────────────────────────────────────

def add_metro_pop(stations: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """
    For each station, find the nearest city within 50 km and assign its
    population as `metro_pop`.  Falls back to nearest city within 100 km,
    then NaN.
    """
    valid = stations[["lat", "lon"]].notna().all(axis=1)
    city_lats = cities["lat"].values
    city_lons = cities["lon"].values
    city_pops = cities["pop"].values

    metro_pop = np.full(len(stations), np.nan)

    for i, row in stations[valid].iterrows():
        dists = haversine_km_vec(row["lat"], row["lon"], city_lats, city_lons)
        close = np.where(dists <= 50.0)[0]
        if len(close) == 0:
            close = np.where(dists <= 100.0)[0]
        if len(close) > 0:
            best = close[np.argmax(city_pops[close])]
            metro_pop[i] = city_pops[best]

    stations = stations.copy()
    stations["metro_pop"] = metro_pop
    return stations


def add_distance_to_major_city(stations: pd.DataFrame,
                                cities: pd.DataFrame) -> pd.DataFrame:
    """Haversine distance (km) to nearest city with population > 500,000."""
    major = cities[cities["pop"] > 500_000].copy()
    if major.empty:
        stations = stations.copy()
        stations["distance_to_nearest_major_city_km"] = np.nan
        return stations

    major_lats = major["lat"].values
    major_lons = major["lon"].values

    valid = stations[["lat", "lon"]].notna().all(axis=1)
    dists_col = np.full(len(stations), np.nan)

    for i, row in stations[valid].iterrows():
        dists = haversine_km_vec(row["lat"], row["lon"], major_lats, major_lons)
        dists_col[i] = dists.min()

    stations = stations.copy()
    stations["distance_to_nearest_major_city_km"] = dists_col
    return stations


def add_num_nearby_stations(stations: pd.DataFrame,
                             radius_km: float = 80.0) -> pd.DataFrame:
    """Count of OTHER Amtrak stations within radius_km of each station."""
    valid_mask = stations[["lat", "lon"]].notna().all(axis=1)
    valid_idx = stations.index[valid_mask].tolist()

    lats = stations.loc[valid_mask, "lat"].values
    lons = stations.loc[valid_mask, "lon"].values

    counts = np.zeros(len(valid_idx), dtype=int)
    for k, i in enumerate(valid_idx):
        dists = haversine_km_vec(lats[k], lons[k], lats, lons)
        # exclude self (dist == 0)
        counts[k] = int((dists <= radius_km).sum()) - 1

    num_nearby = pd.Series(np.nan, index=stations.index)
    for k, i in enumerate(valid_idx):
        num_nearby[i] = counts[k]

    stations = stations.copy()
    stations["num_nearby_stations"] = num_nearby.astype("Int64")
    return stations


def add_is_northeast_corridor(stations: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if station is within the NEC bounding box."""
    stations = stations.copy()
    stations["is_northeast_corridor"] = (
        (stations["lat"] >= NEC_LAT_MIN)
        & (stations["lat"] <= NEC_LAT_MAX)
        & (stations["lon"] >= NEC_LON_MIN)
    ).astype(int)
    return stations


# ── Main ───────────────────────────────────────────────────────────────────────

def compute_all_features(stations: pd.DataFrame,
                          cities: pd.DataFrame) -> pd.DataFrame:
    print("  Computing metro_pop …")
    stations = add_metro_pop(stations, cities)

    print("  Computing distance_to_nearest_major_city_km …")
    stations = add_distance_to_major_city(stations, cities)

    print("  Computing num_nearby_stations …")
    stations = add_num_nearby_stations(stations)

    print("  Computing is_northeast_corridor …")
    stations = add_is_northeast_corridor(stations)

    return stations


def main():
    in_path = PROCESSED / "stations.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found — run parse_and_join.py first"
        )

    print("Loading stations.csv …")
    stations = pd.read_csv(in_path)
    print(f"  {len(stations)} stations")

    print("Loading US cities population data …")
    cities = pd.read_csv(CITIES_CSV)
    print(f"  {len(cities)} cities")

    print("Computing features …")
    stations = compute_all_features(stations, cities)

    stations.to_csv(in_path, index=False)
    print(f"\nUpdated {in_path} with engineered features")

    # Summary
    feature_cols = ["metro_pop", "distance_to_nearest_major_city_km",
                    "num_nearby_stations", "is_northeast_corridor"]
    print(stations[feature_cols].describe().to_string())
    print(f"\nNEC stations: {stations['is_northeast_corridor'].sum()}")


if __name__ == "__main__":
    main()
