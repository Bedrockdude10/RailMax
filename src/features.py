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

from config import (
    METRO_POP_RADIUS_KM,
    METRO_POP_FALLBACK_KM,
    MAJOR_CITY_POP_THRESHOLD,
    NEARBY_STATIONS_RADIUS_KM,
    NEC_LAT_MIN,
    NEC_LAT_MAX,
    NEC_LON_MIN,
)
from utils import haversine_km

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
CITIES_CSV = ROOT / "data" / "us_cities_pop.csv"


# ── Feature functions ──────────────────────────────────────────────────────────

def add_metro_pop(stations: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """
    For each station, find the nearest city within METRO_POP_RADIUS_KM and
    assign its population as `metro_pop`.  Falls back to METRO_POP_FALLBACK_KM,
    then NaN.

    Fully vectorised: builds an (N_stations × N_cities) distance matrix once.
    """
    valid = stations[["lat", "lon"]].notna().all(axis=1)
    s_lats = stations.loc[valid, "lat"].values[:, None]   # (N_s, 1)
    s_lons = stations.loc[valid, "lon"].values[:, None]
    c_lats = cities["lat"].values[None, :]                # (1, N_c)
    c_lons = cities["lon"].values[None, :]
    city_pops = cities["pop"].values                      # (N_c,)

    # (N_s, N_c) distance matrix
    dist_km = haversine_km(s_lats, s_lons, c_lats, c_lons)

    metro_pop = np.full(len(stations), np.nan)
    valid_positions = np.where(valid)[0]

    for k, i in enumerate(valid_positions):
        dists = dist_km[k]
        close = np.where(dists <= METRO_POP_RADIUS_KM)[0]
        if len(close) == 0:
            close = np.where(dists <= METRO_POP_FALLBACK_KM)[0]
        if len(close) > 0:
            best = close[np.argmax(city_pops[close])]
            metro_pop[i] = city_pops[best]

    stations = stations.copy()
    stations["metro_pop"] = metro_pop
    return stations


def add_distance_to_major_city(stations: pd.DataFrame,
                                cities: pd.DataFrame) -> pd.DataFrame:
    """
    Haversine distance (km) to nearest city with population > MAJOR_CITY_POP_THRESHOLD.

    Fully vectorised: builds an (N_stations × N_major_cities) distance matrix once.
    """
    major = cities[cities["pop"] > MAJOR_CITY_POP_THRESHOLD]
    stations = stations.copy()
    if major.empty:
        stations["distance_to_nearest_major_city_km"] = np.nan
        return stations

    valid = stations[["lat", "lon"]].notna().all(axis=1)
    s_lats = stations.loc[valid, "lat"].values[:, None]   # (N_s, 1)
    s_lons = stations.loc[valid, "lon"].values[:, None]
    m_lats = major["lat"].values[None, :]                 # (1, N_major)
    m_lons = major["lon"].values[None, :]

    dist_km = haversine_km(s_lats, s_lons, m_lats, m_lons)  # (N_s, N_major)

    dists_col = np.full(len(stations), np.nan)
    valid_positions = np.where(valid)[0]
    for k, i in enumerate(valid_positions):
        dists_col[i] = dist_km[k].min()

    stations["distance_to_nearest_major_city_km"] = dists_col
    return stations


def add_num_nearby_stations(stations: pd.DataFrame,
                             radius_km: float = NEARBY_STATIONS_RADIUS_KM) -> pd.DataFrame:
    """
    Count of OTHER Amtrak stations within radius_km of each station.

    Fully vectorised: builds an (N_valid × N_valid) pairwise distance matrix once.
    """
    valid_mask = stations[["lat", "lon"]].notna().all(axis=1)
    valid_positions = np.where(valid_mask)[0]

    lats = stations.loc[valid_mask, "lat"].values
    lons = stations.loc[valid_mask, "lon"].values

    # (N_valid, N_valid) pairwise distance matrix
    dist_km = haversine_km(lats[:, None], lons[:, None], lats[None, :], lons[None, :])

    # Count stations within radius, excluding self (diagonal = 0)
    counts = (dist_km <= radius_km).sum(axis=1) - 1

    num_nearby = pd.array([pd.NA] * len(stations), dtype="Int64")
    for k, i in enumerate(valid_positions):
        num_nearby[i] = int(counts[k])

    stations = stations.copy()
    stations["num_nearby_stations"] = num_nearby
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

    feature_cols = ["metro_pop", "distance_to_nearest_major_city_km",
                    "num_nearby_stations", "is_northeast_corridor"]
    print(stations[feature_cols].describe().to_string())
    print(f"\nNEC stations: {stations['is_northeast_corridor'].sum()}")


if __name__ == "__main__":
    main()
