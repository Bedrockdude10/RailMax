"""
data_loader.py

Standardised loader for data/processed/stations.csv.
Returns a clean DataFrame ready for EBM training.

Handles:
- NaN filling for v0 features (IPCD features missing for some stations)
- NaN placeholders for v1 features not yet present (daily_departures, etc.)
- Type coercion
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# ── Feature definitions ────────────────────────────────────────────────────────

# Binary IPCD features — fill NaN with 0 (no data → assume not present)
BINARY_IPCD_FEATURES = [
    "has_heavy_rail",
    "has_commuter_rail",
    "has_light_rail",
    "has_transit_bus",
    "has_intercity_bus",
    "has_air_connection",
    "has_bikeshare",
]

# Numeric features present in v0
NUMERIC_V0_FEATURES = [
    "lat",
    "lon",
    "modes_served",
    "is_metro_area",
    "metro_pop",
    "distance_to_nearest_major_city_km",
    "num_nearby_stations",
    "is_northeast_corridor",
]

# Categorical features
CATEGORICAL_FEATURES = [
    "station_type",
]

# GTFS-derived features (computed by src/build_gtfs_features.py)
GTFS_FEATURES = [
    "daily_departures",
    "num_routes_served",
    "is_terminal",
    "avg_dwell_time_sec",
    "service_span_hours",
    "avg_stop_sequence_pct",
    "num_directions",
    "avg_route_length_km",
    "max_route_length_km",
    "num_weekday_trips",
    "pct_long_distance",
]

# ACS commute mode features (computed by src/build_acs_features.py)
ACS_FEATURES = [
    "pct_drove_alone",
    "pct_public_transit",
    "pct_rail_commute",
    "pct_walked",
    "pct_work_from_home",
    "median_household_income",
]

ALL_FEATURES = BINARY_IPCD_FEATURES + NUMERIC_V0_FEATURES + CATEGORICAL_FEATURES + GTFS_FEATURES + ACS_FEATURES

TARGET = "annual_ridership"

# Hold-out station codes (NTAD Code column).
# WAS = Washington Union Station, BOS = Boston South Station,
# PHL = Philadelphia 30th Street, CHI = Chicago Union Station
HOLDOUT_CODES = {"WAS", "BOS", "PHL", "CHI"}


# ── Loader ────────────────────────────────────────────────────────────────────

def load_stations(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and clean stations.csv.

    Returns a DataFrame with:
    - All v0 features, NaN-filled where appropriate
    - All GTFS features (NaN for stations without GTFS coverage)
    - `log_ridership` column (log1p of annual_ridership)
    - `is_holdout` flag for the four major validation stations
    """
    if path is None:
        path = PROCESSED / "stations.csv"

    df = pd.read_csv(path, low_memory=False)

    # ── Fill binary IPCD features ──
    for col in BINARY_IPCD_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
        else:
            df[col] = 0

    # ── Fill modes_served ──
    if "modes_served" in df.columns:
        df["modes_served"] = df["modes_served"].fillna(0).astype(float)
    else:
        df["modes_served"] = 0.0

    # ── is_metro_area ──
    if "is_metro_area" in df.columns:
        df["is_metro_area"] = df["is_metro_area"].fillna(0).astype(int)
    else:
        df["is_metro_area"] = 0

    # ── Numeric features: leave NaN (EBM handles missing) ──
    for col in ["metro_pop", "distance_to_nearest_major_city_km",
                "num_nearby_stations"]:
        if col not in df.columns:
            df[col] = np.nan

    # ── NEC flag ──
    if "is_northeast_corridor" not in df.columns:
        df["is_northeast_corridor"] = 0
    else:
        df["is_northeast_corridor"] = df["is_northeast_corridor"].fillna(0).astype(int)

    # ── station_type: fill with most common ──
    if "station_type" in df.columns:
        mode_val = df["station_type"].mode()
        df["station_type"] = df["station_type"].fillna(
            mode_val[0] if len(mode_val) > 0 else "unknown"
        )
    else:
        df["station_type"] = "unknown"

    # ── Log-transform target ──
    if TARGET in df.columns:
        df["log_ridership"] = np.log1p(df[TARGET])

    # ── Hold-out flag (matched on station code) ──
    if "code" in df.columns:
        df["is_holdout"] = df["code"].isin(HOLDOUT_CODES).astype(int)
    else:
        df["is_holdout"] = 0

    # ── GTFS features: leave NaN for stations with no GTFS data (EBM handles missing) ──
    for col in GTFS_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # ── ACS features: leave NaN for unmatched stations (EBM handles missing) ──
    for col in ACS_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    return df


def get_feature_matrix(df: pd.DataFrame,
                        features: Optional[list] = None) -> pd.DataFrame:
    """
    Return X (feature matrix) for model training/inference.
    Uses ALL_FEATURES by default; filters to columns present in df.
    """
    if features is None:
        features = ALL_FEATURES
    available = [f for f in features if f in df.columns]
    return df[available].copy()


def get_train_test_split(df: pd.DataFrame):
    """
    Split into train set (is_holdout == 0, annual_ridership not null)
    and holdout set (is_holdout == 1).
    """
    has_ridership = df["annual_ridership"].notna()
    train = df[has_ridership & (df["is_holdout"] == 0)].copy()
    holdout = df[has_ridership & (df["is_holdout"] == 1)].copy()
    return train, holdout


if __name__ == "__main__":
    df = load_stations()
    print(f"Loaded {len(df)} stations")
    print(f"Columns: {df.columns.tolist()}")
    train, holdout = get_train_test_split(df)
    print(f"Train: {len(train)}, Holdout: {len(holdout)}")
    print("\nHoldout stations:")
    name_col = "station_name" if "station_name" in holdout.columns else "Station"
    print(holdout[[name_col, "annual_ridership"]].to_string(index=False))
    print("\nFeature null counts:")
    print(df[ALL_FEATURES].isna().sum().to_string())
