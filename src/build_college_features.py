"""
build_college_features.py

Join IPEDS college enrollment data to stations by proximity, adding three
radius-based features to data/processed/stations.csv.

Sources:
  data/raw/hd2023.csv       — institution directory (UNITID, lat, lon)
  data/raw/ef2023a_rv.csv   — fall 2023 enrollment (revised)
                              EFALEVEL=1, LINE=29 → total headcount per institution

Features added per station:
  college_enrollment_5km   — sum of EFTOTLT for institutions within 5 km
  college_enrollment_15km  — sum of EFTOTLT for institutions within 15 km
  college_enrollment_30km  — sum of EFTOTLT for institutions within 30 km

Run:
    python src/build_college_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

from config import COLLEGE_RADII_KM
from utils import EARTH_RADIUS_KM

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

COLLEGE_FEATURES = [f"college_enrollment_{r}km" for r in COLLEGE_RADII_KM]
RADII_KM = COLLEGE_RADII_KM


# ── Load IPEDS ────────────────────────────────────────────────────────────────

def load_colleges() -> pd.DataFrame:
    """Return DataFrame of colleges with lat, lon, total_enrollment."""
    print("Loading hd2023.csv …")
    hd = pd.read_csv(RAW / "hd2023.csv", encoding="utf-8-sig", low_memory=False,
                     usecols=["UNITID", "INSTNM", "LATITUDE", "LONGITUD"])
    hd = hd.dropna(subset=["LATITUDE", "LONGITUD"])
    print(f"  {len(hd)} institutions with coordinates")

    print("Loading ef2023a_rv.csv …")
    ef = pd.read_csv(RAW / "ef2023a_rv.csv", low_memory=False,
                     usecols=["UNITID", "EFALEVEL", "LINE", "EFTOTLT"])

    # EFALEVEL=1, LINE=29: grand total headcount across all levels
    totals = ef[(ef["EFALEVEL"] == 1) & (ef["LINE"] == 29)][["UNITID", "EFTOTLT"]].copy()
    totals["EFTOTLT"] = pd.to_numeric(totals["EFTOTLT"], errors="coerce").fillna(0)
    print(f"  {len(totals)} enrollment rows; total US enrollment: {totals['EFTOTLT'].sum():,.0f}")

    colleges = hd.merge(totals, on="UNITID", how="left")
    colleges["EFTOTLT"] = colleges["EFTOTLT"].fillna(0)
    print(f"  Merged: {len(colleges)} colleges")
    return colleges


# ── Radius sums ───────────────────────────────────────────────────────────────

def compute_enrollment_within_radii(stations: pd.DataFrame,
                                    colleges: pd.DataFrame) -> pd.DataFrame:
    """
    For each station, sum enrollment of all colleges within each radius.
    Uses vectorised Haversine via sklearn (radians input).
    """
    print(f"\nComputing enrollment within {RADII_KM} km radii …")

    # Convert to radians for sklearn haversine_distances
    s_rad = np.radians(stations[["lat", "lon"]].values)   # (N_stations, 2)
    c_rad = np.radians(colleges[["LATITUDE", "LONGITUD"]].values)  # (N_colleges, 2)

    # Distance matrix in km: shape (N_stations, N_colleges)
    dist_km = haversine_distances(s_rad, c_rad) * EARTH_RADIUS_KM

    enrollment = colleges["EFTOTLT"].values  # (N_colleges,)

    results = {}
    for r in RADII_KM:
        mask = dist_km <= r  # (N_stations, N_colleges) bool
        results[f"college_enrollment_{r}km"] = (mask * enrollment).sum(axis=1)

    feat_df = pd.DataFrame(results, index=stations.index)

    # Diagnostics
    for col in feat_df.columns:
        nonzero = (feat_df[col] > 0).sum()
        print(f"  {col}: {nonzero}/{len(feat_df)} stations with >0 enrollment nearby "
              f"(median {feat_df[col].median():,.0f}, max {feat_df[col].max():,.0f})")

    return feat_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    stations = pd.read_csv(PROCESSED / "stations.csv", low_memory=False)
    print(f"Loaded {len(stations)} stations")

    colleges = load_colleges()

    feat_df = compute_enrollment_within_radii(stations, colleges)

    # Drop stale columns if re-running
    for col in COLLEGE_FEATURES:
        if col in stations.columns:
            stations = stations.drop(columns=[col])

    merged = pd.concat([stations, feat_df], axis=1)

    out_path = PROCESSED / "stations.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nUpdated stations.csv ({len(merged)} rows, {len(merged.columns)} cols)")

    print("\nCollege enrollment feature summary:")
    print(merged[COLLEGE_FEATURES].describe().round(0).to_string())

    print("\nSample (top 8 stations by 15km enrollment):")
    sample_cols = ["station_name"] + COLLEGE_FEATURES
    print(merged.nlargest(8, "college_enrollment_15km")[sample_cols].to_string(index=False))


if __name__ == "__main__":
    main()
