"""
data_loader.py

Standardised loader for data/processed/stations.csv.
Returns a clean DataFrame ready for EBM training.

Handles:
- NaN filling for v0 features (IPCD features missing for some stations)
- NaN placeholders for v1 features not yet present in older datasets
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
    "modes_served",
    "metro_pop",
    "distance_to_nearest_major_city_km",
    "num_amtrak_stations_80km",
]

# Categorical features
CATEGORICAL_FEATURES = [
    "station_type",
]

# GTFS-derived features (computed by src/build_gtfs_features.py)
GTFS_FEATURES = [
    "weekly_departures",
    "is_terminal",
    "avg_dwell_time_sec",
    "service_span_hours",
    "avg_stop_sequence_pct",
    "num_directions",
    "avg_route_length_km",
    "max_route_length_km",
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

# College proximity features (computed by src/build_college_features.py)
COLLEGE_FEATURES = [
    "college_enrollment_15km",
]

# Tourism features (computed by src/build_tourism_features.py)
TOURISM_FEATURES = [
    "overseas_visitors_thousands",
]

ALL_FEATURES = BINARY_IPCD_FEATURES + NUMERIC_V0_FEATURES + CATEGORICAL_FEATURES + GTFS_FEATURES + ACS_FEATURES + COLLEGE_FEATURES + TOURISM_FEATURES

TARGET = "annual_ridership"


# ── Loader ────────────────────────────────────────────────────────────────────

def load_stations(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and clean stations.csv.

    Returns a DataFrame with:
    - All v0 features, NaN-filled where appropriate
    - All GTFS features (NaN for stations without GTFS coverage)
    - `log_ridership` column (log1p of annual_ridership)
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

    # ── Numeric features: leave NaN (EBM handles missing) ──
    for col in ["metro_pop", "distance_to_nearest_major_city_km",
                "num_amtrak_stations_80km"]:
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

    # ── GTFS features: leave NaN for stations with no GTFS data (EBM handles missing) ──
    for col in GTFS_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # ── ACS features: leave NaN for unmatched stations (EBM handles missing) ──
    for col in ACS_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # ── College features: 0 for stations with no colleges nearby ──
    for col in COLLEGE_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    # ── Tourism features: NaN for stations outside top-50 tourist MSAs ──
    for col in TOURISM_FEATURES:
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


if __name__ == "__main__":
    df = load_stations()
    print(f"Loaded {len(df)} stations")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFeature null counts:")
    print(df[ALL_FEATURES].isna().sum().to_string())