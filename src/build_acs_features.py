"""
build_acs_features.py

Parse ACS 5-Year county-level data, reverse-geocode each station's lat/lon to
a county FIPS code via the FCC Census Block API, join features to
data/processed/stations.csv, and write in-place.

Features added per station (from the station's county):
  B08301 — commute mode:
    pct_drove_alone       = B08301_003E / B08301_001E
    pct_public_transit    = B08301_010E / B08301_001E
    pct_rail_commute      = B08301_013E / B08301_001E
    pct_walked            = B08301_019E / B08301_001E
    pct_work_from_home    = B08301_021E / B08301_001E
  B19013 — household income:
    median_household_income = B19013_001E  (dollars, 2023 inflation-adjusted)

FCC API: https://geo.fcc.gov/api/census/area?lat={lat}&lon={lon}&format=json
Results cached in data/processed/station_county_fips.csv so re-runs skip API.

Run:
    python src/build_acs_features.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIPS_CACHE = PROCESSED / "station_county_fips.csv"

B08301_FILE = RAW / "ACSDT5Y2023.B08301-Data.csv"
B19013_FILE = RAW / "ACSDT5Y2023.B19013-Data.csv"

ACS_FEATURES = [
    "pct_drove_alone",
    "pct_public_transit",
    "pct_rail_commute",
    "pct_walked",
    "pct_work_from_home",
    "median_household_income",
]

# B08301 commute mode: feature name → numerator column (denominator = B08301_001E)
B08301_NUMERATORS = {
    "pct_drove_alone":    "B08301_003E",
    "pct_public_transit": "B08301_010E",
    "pct_rail_commute":   "B08301_013E",
    "pct_walked":         "B08301_019E",
    "pct_work_from_home": "B08301_021E",
}


# ── Parse ACS ─────────────────────────────────────────────────────────────────

def _parse_acs_file(path: Path, cols: list[str]) -> pd.DataFrame:
    """
    Generic ACS file parser.
    Row 0: machine-readable column names. Row 1: human labels (skipped).
    Coerces requested columns to numeric; non-numeric (-, N) → NaN.
    Returns df with county_fips + requested cols.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, skiprows=[1])
    df["county_fips"] = df["GEO_ID"].str[-5:]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_acs() -> pd.DataFrame:
    """Load and merge B08301 (commute mode) and B19013 (household income)."""
    # ── B08301 commute mode ──
    print(f"Parsing {B08301_FILE.name} …")
    b08 = _parse_acs_file(
        B08301_FILE,
        ["B08301_001E"] + list(B08301_NUMERATORS.values()),
    )
    total = b08["B08301_001E"]
    for feat, num_col in B08301_NUMERATORS.items():
        b08[feat] = b08[num_col] / total
    commute_features = list(B08301_NUMERATORS.keys())
    b08 = b08[["county_fips"] + commute_features]
    print(f"  {len(b08)} counties, {b08[commute_features].notna().all(axis=1).sum()} complete")

    # ── B19013 median household income ──
    print(f"Parsing {B19013_FILE.name} …")
    b19 = _parse_acs_file(B19013_FILE, ["B19013_001E"])
    b19 = b19[["county_fips", "B19013_001E"]].rename(
        columns={"B19013_001E": "median_household_income"}
    )
    print(f"  {len(b19)} counties, {b19['median_household_income'].notna().sum()} with income data")

    acs = b08.merge(b19, on="county_fips", how="outer")
    return acs


# ── FCC reverse-geocode ───────────────────────────────────────────────────────

FCC_URL = "https://geo.fcc.gov/api/census/area?lat={lat}&lon={lon}&format=json"
RETRY_WAIT = 2   # seconds between retries
MAX_RETRIES = 3


def fcc_county_fips(lat: float, lon: float) -> str | None:
    """Call FCC Census Block API; return 5-digit county FIPS or None on failure."""
    url = FCC_URL.format(lat=lat, lon=lon)
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            results = data.get("results", [])
            if results:
                return results[0].get("county_fips")
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(RETRY_WAIT * (attempt + 1))
            else:
                return None
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
            else:
                return None
    return None


def get_station_fips(stations: pd.DataFrame) -> pd.DataFrame:
    """
    For each station, look up county FIPS via FCC API.
    Results cached in FIPS_CACHE; only uncached stations hit the API.
    Returns DataFrame with columns [code, county_fips].
    """
    # Load cache
    if FIPS_CACHE.exists():
        cache = pd.read_csv(FIPS_CACHE, dtype=str)
        cached_codes = set(cache["code"].tolist())
    else:
        cache = pd.DataFrame(columns=["code", "county_fips"])
        cached_codes = set()

    need = stations[~stations["code"].isin(cached_codes)].copy()
    print(f"  {len(cached_codes)} stations already cached, {len(need)} need API lookup")

    if len(need) == 0:
        return cache

    new_rows = []
    total = len(need)
    for i, (_, row) in enumerate(need.iterrows()):
        code = row["code"]
        lat, lon = row["lat"], row["lon"]

        if pd.isna(lat) or pd.isna(lon):
            fips = None
        else:
            fips = fcc_county_fips(lat, lon)

        new_rows.append({"code": code, "county_fips": fips})

        # Progress indicator every 50 stations
        if (i + 1) % 50 == 0 or (i + 1) == total:
            done = sum(1 for r in new_rows if r["county_fips"] is not None)
            print(f"  [{i+1}/{total}] {done} resolved so far …")

        # Be polite to the API: ~10 req/s
        time.sleep(0.1)

    new_df = pd.DataFrame(new_rows, dtype=str)
    combined = pd.concat([cache, new_df], ignore_index=True)
    combined.to_csv(FIPS_CACHE, index=False)
    print(f"  FIPS cache updated → {FIPS_CACHE}")
    return combined


# ── Join ──────────────────────────────────────────────────────────────────────

def join_acs_to_stations(stations: pd.DataFrame,
                          fips_df: pd.DataFrame,
                          acs: pd.DataFrame) -> pd.DataFrame:
    # Drop stale ACS columns if re-running
    for col in ACS_FEATURES:
        if col in stations.columns:
            stations = stations.drop(columns=[col])

    # fips_df may have "None" strings from CSV — normalise
    fips_df = fips_df.copy()
    fips_df["county_fips"] = fips_df["county_fips"].replace("None", np.nan)

    # Merge station → county_fips → ACS features
    merged = stations.merge(fips_df[["code", "county_fips"]], on="code", how="left")
    merged = merged.merge(acs[["county_fips"] + ACS_FEATURES], on="county_fips", how="left")

    joined = merged[ACS_FEATURES].notna().all(axis=1).sum()
    print(f"  ACS features joined to {joined}/{len(merged)} stations")

    # Print coverage gaps
    missing = merged[merged["county_fips"].isna() | merged[ACS_FEATURES[0]].isna()]
    if len(missing) > 0:
        print(f"  {len(missing)} stations with no ACS match (NaN — EBM handles missing):")
        if len(missing) <= 10:
            print("   ", missing[["code", "station_name"]].to_string(index=False))

    return merged.drop(columns=["county_fips"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    stations = pd.read_csv(PROCESSED / "stations.csv", low_memory=False)
    print(f"Loaded {len(stations)} stations")

    acs = load_acs()

    print(f"\nReverse-geocoding {len(stations)} stations to county FIPS …")
    fips_df = get_station_fips(stations)

    resolved = fips_df["county_fips"].replace("None", np.nan).notna().sum()
    print(f"  Total resolved: {resolved}/{len(fips_df)}")

    print("\nJoining ACS features …")
    merged = join_acs_to_stations(stations, fips_df, acs)

    # Sample check
    print("\nSample (stations with ACS data):")
    sample_cols = ["code", "station_name"] + ACS_FEATURES
    sample = merged[merged["pct_drove_alone"].notna()].head(8)[sample_cols]
    print(sample.round(4).to_string(index=False))

    out_path = PROCESSED / "stations.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nUpdated stations.csv written ({len(merged)} rows, {len(merged.columns)} cols)")

    print("\nACS feature summary:")
    print(merged[ACS_FEATURES].describe().round(4).to_string())


if __name__ == "__main__":
    main()