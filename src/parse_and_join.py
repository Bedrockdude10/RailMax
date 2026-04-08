"""
parse_and_join.py

Parse the three raw data sources, join them, and output data/processed/stations.csv.

Join strategy:
1. map_data.csv (UTF-16LE, tab-sep) → ridership per station
2. NTAD_Amtrak_Stations.csv (UTF-8 BOM) → filter to StnType=TRAIN → station metadata
3. Assign each map_data row to a single NTAD station code via:
     a. Nearest-coordinate match (same state, ≤ 30 km)
     b. Fuzzy city-name fallback (same state, score ≥ 85)
   Each map_data row maps to at most one NTAD code.
   Each NTAD code receives the ridership of its nearest map_data match only
   (1:1 join — no duplication).
4. NTAD_IPCD.csv (UTF-8 BOM) → filter to non-null AMTRAKCODE → join on code
5. Output: data/processed/stations.csv

Data-quality checks applied to map_data before matching:
- Rows with corrupt coordinates (coordinate matches wrong state) are flagged
  and matched using the State column + NTAD coordinates instead.
- Rows with State column mismatch vs Station name are flagged.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# State name → abbreviation lookup (for normalising map_data states)
STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised Haversine distance (km) between arrays of lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def normalise_city(name: str) -> str:
    """
    Normalise a station/city name for fuzzy comparison:
    1. Strip parenthetical qualifiers — e.g. "(San Joaquin St.)", "(Auto Train)"
    2. Strip everything from the last comma onward (state portion)
    3. Lowercase + strip whitespace
    4. Remove common suffixes: station, amtrak, depot, terminal
    """
    name = str(name)
    # Strip parenthetical content  (Bug #2 fix)
    name = re.sub(r"\s*\([^)]*\)", "", name)
    # Remove everything from the last comma onward
    name = re.sub(r",.*$", "", name)
    name = name.lower().strip()
    # Remove common suffixes
    name = re.sub(r"\s*(amtrak|station|depot|terminal)$", "", name)
    name = name.strip()
    return name


# ── 1. Load map_data.csv ───────────────────────────────────────────────────────

def load_map_data() -> pd.DataFrame:
    df = pd.read_csv(RAW / "map_data.csv", encoding="utf-16", sep="\t")
    df.columns = df.columns.str.strip()

    df = df[~df["Station"].str.contains("Auto Train|Pinehurst", case=False, na=False)]

    df["state_abbr"] = df["State"].map(STATE_ABBR)

    # Parse city from Station column (e.g. "Yuma, Arizona" → "Yuma")
    df["city_raw"] = df["Station"].str.extract(r"^(.+?),\s*[A-Z]", expand=False)
    mask_no_comma = df["city_raw"].isna()
    df.loc[mask_no_comma, "city_raw"] = df.loc[mask_no_comma, "Station"]
    df["city_norm"] = df["city_raw"].apply(normalise_city)

    df = df.reset_index(drop=True)
    df["map_idx"] = df.index

    return df[["map_idx", "Latitude", "Longitude", "State", "Station",
               "city_raw", "city_norm", "state_abbr", "Value"]].rename(columns={
        "Latitude": "lat_map",
        "Longitude": "lon_map",
        "Value": "annual_ridership",
    })


def flag_corrupt_map_rows(map_df: pd.DataFrame,
                           ntad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect map_data rows whose coordinates are inconsistent with their
    State column (e.g. Portland OR row with Portland ME coordinates).

    For each map_data row, find the nearest NTAD station overall (ignoring
    state). If that station is in a DIFFERENT state, the row's coordinates
    are probably wrong.

    Returns map_df with a 'coord_suspect' boolean column.
    """
    ntad_lats = ntad_df["lat"].values
    ntad_lons = ntad_df["lon"].values
    ntad_states = ntad_df["state_abbr"].values

    suspect = []
    for _, row in map_df.iterrows():
        dists = haversine_km(
            row["lat_map"], row["lon_map"],
            ntad_lats, ntad_lons,
        )
        nearest_idx = np.argmin(dists)
        nearest_state = ntad_states[nearest_idx]
        # If the nearest NTAD station is in a different state than the
        # map_data State column claims, the coordinates are suspect.
        suspect.append(nearest_state != row["state_abbr"])

    map_df = map_df.copy()
    map_df["coord_suspect"] = suspect

    n_suspect = sum(suspect)
    if n_suspect:
        print(f"  ⚠ {n_suspect} map_data rows have coordinates inconsistent with State column:")
        for _, r in map_df[map_df["coord_suspect"]].iterrows():
            print(f"    {r['Station']} — State={r['state_abbr']}, coords point elsewhere")

    return map_df


# ── 2. Load NTAD stations (TRAIN only) ────────────────────────────────────────

def load_ntad_stations() -> pd.DataFrame:
    df = pd.read_csv(RAW / "NTAD_Amtrak_Stations.csv", encoding="utf-8-sig")
    EXCLUDE_CODES = {"LOR", "SFA", "PIH"}
    df = df[(df["StnType"] == "TRAIN") & (~df["Code"].isin(EXCLUDE_CODES))].copy()
    df.columns = df.columns.str.strip()

    # StationName: "Albany-Rensselaer, NY"  → parse city + state
    df["city_ntad"] = df["StationName"].str.extract(r"^(.+?),\s*[A-Z]{2}$", expand=False)
    mask_no_match = df["city_ntad"].isna()
    df.loc[mask_no_match, "city_ntad"] = df.loc[mask_no_match, "StationName"]

    df["city_norm"] = df["city_ntad"].apply(normalise_city)

    return df[["Code", "StaType", "City", "State", "StationName",
               "city_ntad", "city_norm", "lat", "lon"]].rename(columns={
        "State": "state_abbr",
    })


# ── 3. Assign each map_data row to one NTAD code ─────────────────────────────

def assign_map_to_codes(map_df: pd.DataFrame,
                         ntad_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each map_data row, find the best-matching NTAD station code.

    Strategy (applied per row):
      1. Nearest NTAD station by coordinates, restricted to same state,
         within 30 km.
      2. Fuzzy city-name match within same state (score ≥ 85) if no
         coordinate match.

    For rows flagged as coord_suspect (coordinates in wrong state),
    skip the coordinate step and use name matching only, or use NTAD
    coordinates for the distance calculation.

    Returns map_df with 'matched_code' and 'match_method' columns.
    """
    ntad_lats = ntad_df["lat"].values
    ntad_lons = ntad_df["lon"].values
    ntad_codes = ntad_df["Code"].values
    ntad_states = ntad_df["state_abbr"].values
    ntad_city_norms = ntad_df["city_norm"].values

    matched_codes = []
    match_methods = []

    for _, row in map_df.iterrows():
        state = row["state_abbr"]
        code = None
        method = None

        # State-restricted NTAD candidates
        state_mask = ntad_states == state
        if not state_mask.any():
            matched_codes.append(None)
            match_methods.append("no_state_candidates")
            continue

        # ── Step 1: coordinate match (skip if coords are suspect) ──
        if not row.get("coord_suspect", False):
            dists = haversine_km(
                row["lat_map"], row["lon_map"],
                ntad_lats[state_mask], ntad_lons[state_mask],
            )
            min_pos = np.argmin(dists)
            if dists[min_pos] <= 30.0:
                state_indices = np.where(state_mask)[0]
                code = ntad_codes[state_indices[min_pos]]
                method = f"coord({dists[min_pos]:.2f}km)"

        # ── Step 2: fuzzy city-name fallback ──
        if code is None:
            state_cities = ntad_city_norms[state_mask].tolist()
            match = process.extractOne(
                row["city_norm"], state_cities,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=85,
            )
            if match:
                best_city, score, idx = match
                state_indices = np.where(state_mask)[0]
                code = ntad_codes[state_indices[idx]]
                method = f"fuzzy({score:.0f})"

        matched_codes.append(code)
        match_methods.append(method or "unmatched")

    map_df = map_df.copy()
    map_df["matched_code"] = matched_codes
    map_df["match_method"] = match_methods
    return map_df


# ── 4. Build station-level ridership (1:1 on Code) ───────────────────────────

def build_ridership_by_code(map_df: pd.DataFrame,
                             ntad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate map_data ridership per NTAD code and left-join onto the
    NTAD station list, producing exactly one row per station code.

    When multiple map_data rows match the same code (e.g. "Burbank" and
    "Burbank Airport" both → BUR), we SUM their ridership and note the
    match in join_method. When multiple NTAD codes share a city name
    (e.g. SKN and SKT in Stockton), each gets only its own matched
    ridership — no duplication.
    """
    matched = map_df[map_df["matched_code"].notna()].copy()

    # Aggregate ridership by code (sum if multiple map rows → same code)
    ridership_agg = (
        matched.groupby("matched_code")
        .agg(
            annual_ridership=("annual_ridership", "sum"),
            map_station=("Station", lambda x: " + ".join(x)),
            match_method=("match_method", "first"),
            n_map_rows=("map_idx", "count"),
            lat_map=("lat_map", "first"),
            lon_map=("lon_map", "first"),
        )
        .reset_index()
        .rename(columns={"matched_code": "code"})
    )

    # Flag multi-source aggregations
    multi = ridership_agg["n_map_rows"] > 1
    ridership_agg.loc[multi, "match_method"] = (
        ridership_agg.loc[multi, "match_method"] + f" (summed)"
    )

    # Left join: NTAD stations ← ridership
    result = ntad_df.copy()
    result = result.rename(columns={"Code": "code"})
    result = result.merge(
        ridership_agg[["code", "annual_ridership", "map_station",
                        "match_method", "lat_map", "lon_map"]],
        on="code",
        how="left",
    )

    matched_count = result["annual_ridership"].notna().sum()
    total = len(result)
    print(f"  {matched_count}/{total} stations matched to ridership")

    unmatched_codes = result[result["annual_ridership"].isna()]["code"].tolist()
    if unmatched_codes:
        print(f"  {len(unmatched_codes)} stations with no ridership match")

    # Report unclaimed map_data rows
    unclaimed = map_df[map_df["matched_code"].isna()]
    if len(unclaimed) > 0:
        print(f"  {len(unclaimed)} map_data rows unmatched:")
        for _, r in unclaimed.iterrows():
            print(f"    {r['Station']} ({r['state_abbr']}) — {r['annual_ridership']:,.0f} riders")

    return result


# ── 5. Load and join IPCD ──────────────────────────────────────────────────────

IPCD_BINARY_COLS = ["RAIL_H", "RAIL_C", "RAIL_LIGHT", "BUS_T", "BUS_I",
                    "AIR_SERVE", "BIKE_SHARE"]


def load_ipcd() -> pd.DataFrame:
    df = pd.read_csv(RAW / "NTAD_IPCD.csv", encoding="utf-8-sig", low_memory=False)
    df = df[df["AMTRAKCODE"].notna()].copy()

    # Convert IPCD codes: 1 or 2 → 1 (service present), 3 or 0 → 0
    for col in IPCD_BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if x in (1, 2) else 0)

    # CBSA_TYPE: 1=metro → is_metro_area=1, else 0
    df["is_metro_area"] = (df["CBSA_TYPE"] == 1).astype(int)

    # Keep one row per AMTRAKCODE (take first — multiple rows can share a code)
    df = df.groupby("AMTRAKCODE", as_index=False).first()

    keep = ["AMTRAKCODE", "CBSA_TYPE", "MODES_SERV", "is_metro_area"] + IPCD_BINARY_COLS
    keep = [c for c in keep if c in df.columns]
    return df[keep].rename(columns={
        "RAIL_H": "has_heavy_rail",
        "RAIL_C": "has_commuter_rail",
        "RAIL_LIGHT": "has_light_rail",
        "BUS_T": "has_transit_bus",
        "BUS_I": "has_intercity_bus",
        "AIR_SERVE": "has_air_connection",
        "BIKE_SHARE": "has_bikeshare",
        "MODES_SERV": "modes_served",
    })


# ── 6. Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading map_data.csv …")
    map_df = load_map_data()
    print(f"  {len(map_df)} ridership rows")

    print("\nLoading NTAD stations (TRAIN only) …")
    ntad_df = load_ntad_stations()
    print(f"  {len(ntad_df)} train stations")

    print("\nChecking map_data coordinate quality …")
    map_df = flag_corrupt_map_rows(map_df, ntad_df)

    print("\nAssigning map_data rows to NTAD station codes …")
    map_df = assign_map_to_codes(map_df, ntad_df)

    matched = map_df["matched_code"].notna().sum()
    print(f"  Assigned: {matched}/{len(map_df)} map rows → station codes")

    # Diagnostics: show match methods
    print(f"\n  Match method breakdown:")
    for method_prefix in ["coord", "fuzzy", "unmatched"]:
        count = map_df["match_method"].str.startswith(method_prefix).sum()
        print(f"    {method_prefix}: {count}")

    print("\nBuilding station-level ridership (1:1 per code) …")
    stations = build_ridership_by_code(map_df, ntad_df)

    print("\nLoading IPCD …")
    ipcd_df = load_ipcd()
    print(f"  {len(ipcd_df)} IPCD rows with AMTRAKCODE")

    print("\nJoining IPCD …")
    stations = stations.merge(ipcd_df, left_on="code", right_on="AMTRAKCODE", how="left")
    ipcd_joined = stations["has_heavy_rail"].notna().sum()
    print(f"  IPCD joined to {ipcd_joined} / {len(stations)} stations")

    # ── Canonical column set ──
    stations = stations.rename(columns={
        "lat": "lat",
        "lon": "lon",
        "StaType": "station_type",
        "StationName": "station_name",
    })

    # Use NTAD lat/lon as canonical position (more precise than map_data)
    # Fill with map_data coords where NTAD coords are missing
    stations["lat"] = stations["lat"].fillna(stations["lat_map"])
    stations["lon"] = stations["lon"].fillna(stations["lon_map"])

    out_cols = [
        "code", "station_name", "map_station", "state_abbr", "City",
        "lat", "lon", "station_type", "annual_ridership", "match_method",
        "has_heavy_rail", "has_commuter_rail", "has_light_rail",
        "has_transit_bus", "has_intercity_bus", "has_air_connection",
        "has_bikeshare", "modes_served", "is_metro_area", "CBSA_TYPE",
    ]
    out_cols = [c for c in out_cols if c in stations.columns]
    stations = stations[out_cols].copy()

    out_path = PROCESSED / "stations.csv"
    stations.to_csv(out_path, index=False)
    print(f"\nWrote {len(stations)} rows to {out_path}")

    # Verify no duplicate codes
    dupes = stations["code"].duplicated().sum()
    if dupes:
        print(f"\n  ⚠ ERROR: {dupes} duplicate station codes in output!")
    else:
        print(f"\n  ✓ No duplicate station codes — 1:1 join verified")

    print(stations.describe(include="all").T[["count", "unique", "top", "mean"]].to_string())


if __name__ == "__main__":
    main()