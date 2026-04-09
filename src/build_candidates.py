"""
build_candidates.py

Generate Network Expansion Candidate rows for the top-100 US cities by
population that have no existing Amtrak TRAIN station.

For each candidate city the following features are populated:
  - IPCD connectivity (has_intercity_bus, has_transit_bus, has_heavy_rail, etc.)
      → coordinate lookup against NTAD_IPCD.csv (max 30 km), non-Amtrak rows only
  - metro_pop, distance_to_nearest_major_city_km, num_amtrak_stations_80km
      → reused from features.py (candidates appended, then stripped back out)
  - ACS commute / income features
      → FCC reverse-geocode → county FIPS → ACS join
      → uses the same FIPS cache as build_acs_features.py
  - college_enrollment_15km
      → coordinate lookup against IPEDS data
  - overseas_visitors_thousands
      → county FIPS → OMB area → tourism xlsx (same logic as add_tourism_features.py)

GTFS rail features (daily_departures, num_routes_served, etc.) are left as NaN —
these cities have no Amtrak service, which is the point.

Output: data/processed/expansion_candidates.csv

Run after the full pipeline has been built (stations.csv must exist):
    python src/build_candidates.py
"""

import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

CITIES_CSV = ROOT / "data" / "us_cities_pop.csv"
IPCD_CSV = RAW / "NTAD_IPCD.csv"
B08301_FILE = RAW / "ACSDT5Y2023.B08301-Data.csv"
B19013_FILE = RAW / "ACSDT5Y2023.B19013-Data.csv"
FIPS_CACHE = PROCESSED / "station_county_fips.csv"
OMB_PATH = RAW / "list1_2023.xlsx"
TOURISM_PATH = RAW / "2024-Top-States-and-Cities-Visited.xlsx"
COLLEGE_HD = RAW / "hd2023.csv"
COLLEGE_EF = RAW / "ef2023a_rv.csv"

IPCD_LOOKUP_RADIUS_KM = 30.0
TOP_N_CITIES = 100
EARTH_RADIUS_KM = 6371.0

# ── Reuse constants from config ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MAJOR_CITY_POP_THRESHOLD,
    METRO_POP_RADIUS_KM,
    METRO_POP_FALLBACK_KM,
    NEARBY_STATIONS_RADIUS_KM,
    COLLEGE_RADII_KM,
)
from utils import haversine_km
from features import add_metro_pop, add_distance_to_major_city, add_num_amtrak_stations_80km
from build_acs_features import load_acs, fcc_county_fips, join_acs_to_stations


# ── 1. Identify candidate cities ──────────────────────────────────────────────

def get_candidate_cities(top_n: int = TOP_N_CITIES) -> pd.DataFrame:
    """
    Return cities from us_cities_pop.csv (top N by population) that have no
    existing Amtrak TRAIN station in stations.csv.
    """
    cities = pd.read_csv(CITIES_CSV)
    cities["name_clean"] = cities["name"].str.strip().str.lower()
    cities = cities.drop_duplicates(subset="name_clean").reset_index(drop=True)

    stations = pd.read_csv(PROCESSED / "stations.csv", low_memory=False)
    station_cities = set(stations["City"].str.strip().str.lower().dropna())

    top = cities.head(top_n).copy()
    candidates = top[~top["name_clean"].isin(station_cities)].copy()
    candidates = candidates.reset_index(drop=True)

    # Assign a synthetic code for cache keying: "CAND_<NAME>"
    candidates["code"] = "CAND_" + candidates["name"].str.strip().str.replace(
        r"[^A-Za-z0-9]", "_", regex=True
    )
    candidates["name"] = candidates["name"].str.strip()
    print(f"  {len(candidates)} candidate cities from top {top_n}")
    return candidates


# ── 2. IPCD connectivity features ─────────────────────────────────────────────

IPCD_BINARY_COLS = {
    "RAIL_H":     "has_heavy_rail",
    "RAIL_C":     "has_commuter_rail",
    "RAIL_LIGHT": "has_light_rail",
    "BUS_T":      "has_transit_bus",
    "BUS_I":      "has_intercity_bus",
    "AIR_SERVE":  "has_air_connection",
    "BIKE_SHARE": "has_bikeshare",
}


def add_ipcd_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    For each candidate city, aggregate IPCD connectivity features from all
    non-Amtrak IPCD rows within IPCD_LOOKUP_RADIUS_KM.

    Uses max aggregation: if any nearby IPCD entry has the service, the city
    is considered to have it. `modes_served` is the count of present services.
    """
    print(f"Loading NTAD_IPCD.csv …")
    ipcd = pd.read_csv(IPCD_CSV, encoding="utf-8-sig", low_memory=False)
    ipcd_valid = ipcd[
        ipcd["LATITUDE"].notna() & ipcd["LONGITUDE"].notna() &
        ipcd["AMTRAKCODE"].isna()   # non-Amtrak rows only
    ].copy()

    # Normalise IPCD codes → binary (1/2 = present, else 0)
    for raw_col in IPCD_BINARY_COLS:
        if raw_col in ipcd_valid.columns:
            ipcd_valid[raw_col] = ipcd_valid[raw_col].apply(
                lambda x: 1 if x in (1, 2) else 0
            )
        else:
            ipcd_valid[raw_col] = 0

    ipcd_lats = ipcd_valid["LATITUDE"].values
    ipcd_lons = ipcd_valid["LONGITUDE"].values

    results = []
    for _, row in candidates.iterrows():
        dists = haversine_km(row["lat"], row["lon"], ipcd_lats, ipcd_lons)
        nearby = ipcd_valid[dists <= IPCD_LOOKUP_RADIUS_KM]

        feat = {}
        for raw_col, feat_name in IPCD_BINARY_COLS.items():
            feat[feat_name] = int(nearby[raw_col].max()) if len(nearby) else 0

        # modes_served: count of services present (excludes Amtrak — not here yet)
        feat["modes_served"] = sum(feat.values())
        feat["is_metro_area"] = 1   # all top-100 cities are metro areas
        feat["CBSA_TYPE"] = 1
        results.append(feat)

    feat_df = pd.DataFrame(results, index=candidates.index)
    return pd.concat([candidates, feat_df], axis=1)


# ── 3. Engineered geo features ────────────────────────────────────────────────

def add_geo_features(candidates: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metro_pop, distance_to_nearest_major_city_km, num_amtrak_stations_80km.

    Strategy: append candidates to existing stations, run the feature functions
    (which compute against the full station pool), then keep only the candidate rows.
    """
    cities = pd.read_csv(CITIES_CSV)
    cities["name"] = cities["name"].str.strip()

    # Append candidates (with dummy cols to satisfy feature functions)
    combined = pd.concat([
        stations[["code", "lat", "lon"]],
        candidates[["code", "lat", "lon"]],
    ], ignore_index=True)

    print("  Computing metro_pop …")
    combined = add_metro_pop(combined, cities)

    print("  Computing distance_to_nearest_major_city_km …")
    combined = add_distance_to_major_city(combined, cities)

    print("  Computing num_amtrak_stations_80km …")
    # Pass only existing Amtrak stations so candidates don't count each other
    combined_for_nearby = pd.concat([
        stations[["code", "lat", "lon"]],
        candidates[["code", "lat", "lon"]],
    ], ignore_index=True)

    # Manually count nearby *existing* Amtrak stations for each candidate
    s_lats = stations["lat"].dropna().values
    s_lons = stations["lon"].dropna().values

    nearby_counts = []
    for _, row in candidates.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            nearby_counts.append(pd.NA)
        else:
            dists = haversine_km(row["lat"], row["lon"], s_lats, s_lons)
            nearby_counts.append(int((dists <= NEARBY_STATIONS_RADIUS_KM).sum()))

    # Pull metro_pop and distance from combined (candidate rows are at the end)
    n_stations = len(stations)
    cand_slice = combined.iloc[n_stations:].reset_index(drop=True)

    candidates = candidates.copy()
    candidates["metro_pop"] = cand_slice["metro_pop"].values
    candidates["distance_to_nearest_major_city_km"] = cand_slice["distance_to_nearest_major_city_km"].values
    candidates["num_amtrak_stations_80km"] = nearby_counts

    return candidates


# ── 4. ACS features ───────────────────────────────────────────────────────────

FCC_URL = "https://geo.fcc.gov/api/census/area?lat={lat}&lon={lon}&format=json"

def add_acs_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Reverse-geocode each candidate to a county FIPS via FCC API (cached),
    then join ACS commute/income features.
    """
    acs = load_acs()

    # Load or init the FIPS cache
    if FIPS_CACHE.exists():
        cache = pd.read_csv(FIPS_CACHE, dtype=str)
    else:
        cache = pd.DataFrame(columns=["code", "county_fips"])

    cached_codes = set(cache["code"].tolist())
    need = candidates[~candidates["code"].isin(cached_codes)]

    if len(need) > 0:
        print(f"  FCC API lookup for {len(need)} candidate cities …")
        new_rows = []
        for i, (_, row) in enumerate(need.iterrows()):
            fips = None
            if not (pd.isna(row["lat"]) or pd.isna(row["lon"])):
                url = FCC_URL.format(lat=row["lat"], lon=row["lon"])
                try:
                    with urllib.request.urlopen(url, timeout=10) as resp:
                        data = json.loads(resp.read())
                    results = data.get("results", [])
                    if results:
                        fips = results[0].get("county_fips")
                except Exception:
                    pass
                time.sleep(0.1)
            new_rows.append({"code": row["code"], "county_fips": fips})
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(need)}] done …")

        new_df = pd.DataFrame(new_rows, dtype=str)
        cache = pd.concat([cache, new_df], ignore_index=True)
        cache.to_csv(FIPS_CACHE, index=False)
        print(f"  FIPS cache updated")
    else:
        print(f"  All {len(candidates)} candidates already in FIPS cache")

    candidates = join_acs_to_stations(candidates, cache, acs)
    return candidates


# ── 5. College enrollment features ───────────────────────────────────────────

def add_college_features(candidates: pd.DataFrame) -> pd.DataFrame:
    if not COLLEGE_HD.exists() or not COLLEGE_EF.exists():
        print("  College data not found — skipping")
        for r in COLLEGE_RADII_KM:
            candidates[f"college_enrollment_{r}km"] = np.nan
        return candidates

    hd = pd.read_csv(COLLEGE_HD, encoding="utf-8-sig", low_memory=False,
                     usecols=["UNITID", "LATITUDE", "LONGITUD"])
    hd = hd.dropna(subset=["LATITUDE", "LONGITUD"])

    ef = pd.read_csv(COLLEGE_EF, low_memory=False,
                     usecols=["UNITID", "EFALEVEL", "LINE", "EFTOTLT"])
    totals = ef[(ef["EFALEVEL"] == 1) & (ef["LINE"] == 29)][["UNITID", "EFTOTLT"]].copy()
    totals["EFTOTLT"] = pd.to_numeric(totals["EFTOTLT"], errors="coerce").fillna(0)

    colleges = hd.merge(totals, on="UNITID", how="left")
    colleges["EFTOTLT"] = colleges["EFTOTLT"].fillna(0)

    s_rad = np.radians(candidates[["lat", "lon"]].values)
    c_rad = np.radians(colleges[["LATITUDE", "LONGITUD"]].values)
    dist_km = haversine_distances(s_rad, c_rad) * EARTH_RADIUS_KM
    enrollment = colleges["EFTOTLT"].values

    candidates = candidates.copy()
    for r in COLLEGE_RADII_KM:
        mask = dist_km <= r
        candidates[f"college_enrollment_{r}km"] = (mask * enrollment).sum(axis=1)

    return candidates


# ── 6. Tourism features ───────────────────────────────────────────────────────

TOURISM_NAME_OVERRIDES = {
    "Washington (DC Metro Area), DC-MD-VA": 47764,
    "Hilo (Big Island), HI MICRO": 25900,
    "Edison-New Brunswick, NJ MD": 29484,
    "Barnstable Town (Cape Cod), MA MSA": 12700,
    "Kapaa (Kauai), HI MICRO": 28180,
}


def add_tourism_features(candidates: pd.DataFrame) -> pd.DataFrame:
    if not OMB_PATH.exists() or not TOURISM_PATH.exists():
        print("  Tourism data not found — skipping")
        candidates["overseas_visitors_thousands"] = np.nan
        return candidates

    # Load OMB delineation
    omb = pd.read_excel(OMB_PATH, header=2)
    omb = omb[pd.to_numeric(omb["CBSA Code"], errors="coerce").notna()].copy()
    omb["county_fips"] = (
        omb["FIPS State Code"].fillna(0).astype(int).astype(str).str.zfill(2)
        + omb["FIPS County Code"].fillna(0).astype(int).astype(str).str.zfill(3)
    )
    omb["area_code"] = omb["Metropolitan Division Code"].fillna(omb["CBSA Code"])
    omb["area_title"] = omb["Metropolitan Division Title"].fillna(omb["CBSA Title"])
    county_map = (
        omb[["county_fips", "area_code", "area_title"]]
        .dropna(subset=["county_fips", "area_code"])
        .drop_duplicates("county_fips")
    )
    county_map = county_map[pd.to_numeric(county_map["area_code"], errors="coerce").notna()]
    county_map["area_code"] = pd.to_numeric(county_map["area_code"]).astype(int)

    # Load tourism
    raw = pd.read_excel(TOURISM_PATH, sheet_name="Cities Visited-95CL", header=None)
    df = raw.iloc[5:123, [1, 3]].copy()
    df.columns = ["msa_name", "visitors_k"]
    df["visitors_k"] = pd.to_numeric(df["visitors_k"], errors="coerce")
    df = df.dropna(subset=["msa_name", "visitors_k"]).nlargest(50, "visitors_k")

    omb_titles = county_map[["area_code", "area_title"]].drop_duplicates("area_code")
    omb_title_list = omb_titles["area_title"].tolist()
    omb_code_list = omb_titles["area_code"].tolist()

    def extract_primary_city(name):
        name = re.sub(r"\s+(MSA|MD|MICRO)\s*$", "", str(name))
        name = re.sub(r",?\s+[A-Z]{2}(?:-[A-Z]{2})*\s*$", "", name)
        return name.split("-")[0].strip()

    def extract_state(name):
        m = re.search(r"([A-Z]{2})(?:-[A-Z]{2})*\s*(?:MSA|MD|MICRO|,)?\s*$", str(name))
        return m.group(1) if m else ""

    area_visitors = {}
    for _, row in df.iterrows():
        tname = str(row["msa_name"])
        visitors = row["visitors_k"]
        if tname in TOURISM_NAME_OVERRIDES:
            code = TOURISM_NAME_OVERRIDES[tname]
            if code is not None:
                area_visitors[code] = area_visitors.get(code, 0) + visitors
            continue
        pcity = extract_primary_city(tname).lower()
        state = extract_state(tname)
        for i, otitle in enumerate(omb_title_list):
            if pcity in otitle.lower() and (not state or state in otitle):
                area_visitors[omb_code_list[i]] = area_visitors.get(omb_code_list[i], 0) + visitors
                break

    # Join candidates → county FIPS → OMB area → tourism
    fips_df = pd.read_csv(FIPS_CACHE, dtype=str)
    fips_df["county_fips"] = fips_df["county_fips"].replace("None", pd.NA)
    merged = candidates.merge(fips_df[["code", "county_fips"]], on="code", how="left")
    merged = merged.merge(county_map[["county_fips", "area_code"]], on="county_fips", how="left")
    candidates = candidates.copy()
    candidates["overseas_visitors_thousands"] = (
        merged["area_code"].map(area_visitors).values
    )
    return candidates


# ── 7. Assemble final output ──────────────────────────────────────────────────

def build_candidate_row(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Rename and fill columns to match the stations.csv schema expected by
    data_loader.py. GTFS rail features are explicitly NaN.
    """
    GTFS_FEATURES = [
        "weekly_departures", "num_routes_served", "is_terminal",
        "avg_dwell_time_sec", "service_span_hours", "avg_stop_sequence_pct",
        "num_directions", "avg_route_length_km", "max_route_length_km",
        "pct_long_distance",
    ]

    df = candidates.rename(columns={"name": "City"}).copy()
    df["station_name"] = df["City"] + " (Expansion Candidate)"
    df["station_type"] = "Expansion Candidate"
    df["annual_ridership"] = np.nan
    df["is_northeast_corridor"] = 0
    df["is_expansion_candidate"] = 1

    for col in GTFS_FEATURES:
        df[col] = np.nan

    # Drop intermediate columns
    df = df.drop(columns=["name_clean", "pop"], errors="ignore")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for path in [PROCESSED / "stations.csv", CITIES_CSV, IPCD_CSV,
                 B08301_FILE, B19013_FILE]:
        if not path.exists():
            sys.exit(f"Error: {path} not found — run full pipeline first")

    stations = pd.read_csv(PROCESSED / "stations.csv", low_memory=False)

    print("Identifying candidate cities …")
    candidates = get_candidate_cities()

    print("\nAdding IPCD connectivity features …")
    candidates = add_ipcd_features(candidates)

    print("\nAdding geo features (metro_pop, distance_to_major, num_amtrak_stations_80km) …")
    candidates = add_geo_features(candidates, stations)

    print("\nAdding ACS features …")
    candidates = add_acs_features(candidates)

    print("\nAdding college enrollment features …")
    candidates = add_college_features(candidates)

    print("\nAdding tourism features …")
    candidates = add_tourism_features(candidates)

    print("\nAssembling final output …")
    candidates = build_candidate_row(candidates)

    out_path = PROCESSED / "expansion_candidates.csv"
    candidates.to_csv(out_path, index=False)
    print(f"\nWrote {len(candidates)} expansion candidates to {out_path}")
    print(f"Columns: {candidates.columns.tolist()}")

    print("\nConnectivity summary:")
    conn_cols = ["City", "has_intercity_bus", "has_transit_bus", "has_heavy_rail",
                 "has_light_rail", "has_air_connection", "modes_served",
                 "metro_pop", "num_amtrak_stations_80km"]
    print(candidates[conn_cols].sort_values("metro_pop", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
