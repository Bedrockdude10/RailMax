"""
parse_and_join.py

Parse the three raw data sources, join them, and output data/processed/stations.csv.

Join strategy:
1. map_data.csv (UTF-16LE, tab-sep) → ridership per station
2. NTAD_Amtrak_Stations.csv (UTF-8 BOM) → filter to StnType=TRAIN → station metadata
3. Join ridership to stations on city name + state abbreviation (fuzzy match,
   nearest-coordinate fallback)
4. NTAD_IPCD.csv (UTF-8 BOM) → filter to non-null AMTRAKCODE → join on AMTRAKCODE=Code
5. Output: data/processed/stations.csv

FIX: coordinate fallback is restricted to same-state candidates only (≤ 30 km),
preventing cross-state mismatches (e.g. Lindenwold NJ grabbing Philadelphia PA
ridership at 20 km). Unclaimed-row tracking prevents double-assignment.
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
    Strip trailing state from strings like 'Albany, NY' or 'Albany, New York'
    and lowercase + strip punctuation.
    """
    name = str(name)
    # Remove everything from the last comma onward
    name = re.sub(r",.*$", "", name)
    name = name.lower().strip()
    # Remove common suffixes like "station", "amtrak"
    name = re.sub(r"\s*(amtrak|station|depot|terminal)$", "", name)
    name = name.strip()
    return name


# ── 1. Load map_data.csv ───────────────────────────────────────────────────────

def load_map_data() -> pd.DataFrame:
    df = pd.read_csv(RAW / "map_data.csv", encoding="utf-16", sep="\t")
    df.columns = df.columns.str.strip()

    # Station column: "City, State Full Name"  e.g. "Yuma, Arizona"
    df["city_raw"] = df["Station"].str.extract(r"^(.+?),\s*[A-Z]", expand=False)
    # If no comma, the whole thing is the city
    mask_no_comma = df["city_raw"].isna()
    df.loc[mask_no_comma, "city_raw"] = df.loc[mask_no_comma, "Station"]

    df["state_abbr"] = df["State"].map(STATE_ABBR)
    df["city_norm"] = df["city_raw"].apply(normalise_city)

    # Add a unique index so we can track which map_data rows are claimed
    df = df.reset_index(drop=True)
    df["map_idx"] = df.index

    return df[["map_idx", "Latitude", "Longitude", "State", "Station",
               "city_raw", "city_norm", "state_abbr", "Value"]].rename(columns={
        "Latitude": "lat_map",
        "Longitude": "lon_map",
        "Value": "annual_ridership",
    })


# ── 2. Load NTAD stations (TRAIN only) ────────────────────────────────────────

def load_ntad_stations() -> pd.DataFrame:
    df = pd.read_csv(RAW / "NTAD_Amtrak_Stations.csv", encoding="utf-8-sig")
    df = df[df["StnType"] == "TRAIN"].copy()
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


# ── 3. Join ridership to NTAD stations ────────────────────────────────────────

def join_ridership_to_stations(map_df: pd.DataFrame,
                                ntad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Primary join: city_norm + state_abbr exact match.
    Secondary: fuzzy city match within same state (score ≥ 85).
    Tertiary: nearest-coordinate fallback (≤ 30 km), but only for
              map_data rows NOT already claimed by exact/fuzzy matches.

    Each map_data ridership row is assigned to at most one NTAD station.
    """
    # Track which map_data rows have been claimed
    claimed_map_idxs = set()

    # ── exact match ──
    exact = ntad_df.merge(
        map_df[["map_idx", "city_norm", "state_abbr", "lat_map", "lon_map",
                "annual_ridership", "Station"]],
        on=["city_norm", "state_abbr"],
        how="left",
    )
    matched = exact["annual_ridership"].notna()
    unmatched_ntad = exact[~matched].copy()
    result = exact[matched].copy()
    result["join_method"] = "exact"

    # Record claimed map_data rows
    claimed_map_idxs.update(result["map_idx"].dropna().astype(int).tolist())

    print(f"  Exact matches: {matched.sum()} / {len(ntad_df)}")

    if len(unmatched_ntad) == 0:
        return result

    # ── fuzzy match within state ──
    fuzzy_rows = []
    for _, row in unmatched_ntad.iterrows():
        state_candidates = map_df[
            (map_df["state_abbr"] == row["state_abbr"]) &
            (~map_df["map_idx"].isin(claimed_map_idxs))
        ]
        if state_candidates.empty:
            continue
        choices = state_candidates["city_norm"].tolist()
        match = process.extractOne(row["city_norm"], choices,
                                   scorer=fuzz.token_sort_ratio,
                                   score_cutoff=85)
        if match:
            best_city, score, idx = match
            candidate = state_candidates[state_candidates["city_norm"] == best_city].iloc[0]
            r = row.to_dict()
            r.update({
                "map_idx": candidate["map_idx"],
                "lat_map": candidate["lat_map"],
                "lon_map": candidate["lon_map"],
                "annual_ridership": candidate["annual_ridership"],
                "Station": candidate["Station"],
                "join_method": f"fuzzy({score:.0f})",
            })
            fuzzy_rows.append(r)
            claimed_map_idxs.add(int(candidate["map_idx"]))

    if fuzzy_rows:
        fuzzy_df = pd.DataFrame(fuzzy_rows)
        fuzzy_matched_codes = set(fuzzy_df["Code"]) if "Code" in fuzzy_df.columns else set()
        still_unmatched = unmatched_ntad[~unmatched_ntad["Code"].isin(fuzzy_matched_codes)].copy()
        result = pd.concat([result, fuzzy_df], ignore_index=True)
        print(f"  Fuzzy matches: {len(fuzzy_rows)}")
    else:
        still_unmatched = unmatched_ntad.copy()

    # ── coordinate fallback (only unclaimed map_data rows, same state) ──
    coord_rows = []
    unclaimed_map = map_df[~map_df["map_idx"].isin(claimed_map_idxs)]

    for _, row in still_unmatched.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue

        # Restrict to same state so we never pull ridership across state lines
        state_unclaimed = unclaimed_map[unclaimed_map["state_abbr"] == row["state_abbr"]]
        if state_unclaimed.empty:
            continue

        dists = haversine_km(row["lat"], row["lon"],
                             state_unclaimed["lat_map"].values,
                             state_unclaimed["lon_map"].values)
        min_pos = np.argmin(dists)
        if dists[min_pos] <= 30.0:
            candidate = state_unclaimed.iloc[min_pos]
            r = row.to_dict()
            r.update({
                "map_idx": candidate["map_idx"],
                "lat_map": candidate["lat_map"],
                "lon_map": candidate["lon_map"],
                "annual_ridership": candidate["annual_ridership"],
                "Station": candidate["Station"],
                "join_method": f"coord({dists[min_pos]:.1f}km)",
            })
            coord_rows.append(r)
            claimed_map_idxs.add(int(candidate["map_idx"]))
            # Update unclaimed set (both global and local views)
            unclaimed_map = unclaimed_map[unclaimed_map["map_idx"] != candidate["map_idx"]]

    if coord_rows:
        coord_df = pd.DataFrame(coord_rows)
        result = pd.concat([result, coord_df], ignore_index=True)
        print(f"  Coordinate fallback matches: {len(coord_rows)}")

        # Print high-ridership coordinate matches for review
        high_rider_coords = coord_df[coord_df["annual_ridership"] > 100_000]
        if len(high_rider_coords) > 0:
            print(f"\n  ⚠ High-ridership coordinate fallback matches (review these):")
            for _, r in high_rider_coords.iterrows():
                code = r.get("Code", "???")
                stn = r.get("StationName", r.get("city_ntad", "???"))
                matched_to = r.get("Station", "???")
                rider = int(r["annual_ridership"])
                method = r["join_method"]
                print(f"    {code} {stn} → {matched_to} ({rider:,} riders) [{method}]")

    no_ridership = len(ntad_df) - len(result)
    print(f"  Stations with no ridership match: {no_ridership}")

    # Drop the map_idx column before returning
    if "map_idx" in result.columns:
        result = result.drop(columns=["map_idx"])

    return result


# ── 4. Load and join IPCD ──────────────────────────────────────────────────────

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


# ── 5. Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading map_data.csv …")
    map_df = load_map_data()
    print(f"  {len(map_df)} ridership rows")

    print("\nLoading NTAD stations (TRAIN only) …")
    ntad_df = load_ntad_stations()
    print(f"  {len(ntad_df)} train stations")

    print("\nJoining ridership to stations …")
    stations = join_ridership_to_stations(map_df, ntad_df)

    print("\nLoading IPCD …")
    ipcd_df = load_ipcd()
    print(f"  {len(ipcd_df)} IPCD rows with AMTRAKCODE")

    print("\nJoining IPCD …")
    stations = stations.merge(ipcd_df, left_on="Code", right_on="AMTRAKCODE", how="left")
    ipcd_joined = stations["has_heavy_rail"].notna().sum()
    print(f"  IPCD joined to {ipcd_joined} / {len(stations)} stations")

    # ── Canonical column set ──
    stations = stations.rename(columns={
        "lat": "lat",
        "lon": "lon",
        "StaType": "station_type",
        "StationName": "station_name",
        "Code": "code",
    })

    # Use NTAD lat/lon as canonical position (more precise than map_data)
    # Fill with map_data coords where NTAD coords are missing
    stations["lat"] = stations["lat"].fillna(stations["lat_map"])
    stations["lon"] = stations["lon"].fillna(stations["lon_map"])

    out_cols = [
        "code", "station_name", "Station", "state_abbr", "City",
        "lat", "lon", "station_type", "annual_ridership", "join_method",
        "has_heavy_rail", "has_commuter_rail", "has_light_rail",
        "has_transit_bus", "has_intercity_bus", "has_air_connection",
        "has_bikeshare", "modes_served", "is_metro_area", "CBSA_TYPE",
    ]
    out_cols = [c for c in out_cols if c in stations.columns]
    stations = stations[out_cols].copy()

    out_path = PROCESSED / "stations.csv"
    stations.to_csv(out_path, index=False)
    print(f"\nWrote {len(stations)} rows to {out_path}")
    print(stations.describe(include="all").T[["count", "unique", "top", "mean"]].to_string())


if __name__ == "__main__":
    main()