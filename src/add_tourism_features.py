"""
add_tourism_feature.py

Add `overseas_visitors_thousands` to data/processed/stations.csv by joining:
  station → county FIPS → OMB delineation (MSA/MD) → tourism visitor counts.

Fully deterministic — no API calls. Uses the county FIPS codes already
computed by build_acs_features.py and the official OMB CBSA delineation file
to assign each station to its MSA or Metropolitan Division.

Prerequisites:
  - data/processed/stations.csv (with county FIPS from build_acs step)
  - data/processed/station_county_fips.csv (from build_acs step)
  - data/raw/list1_2023.xlsx (OMB CBSA delineation file, Jul 2023)
  - data/raw/tourism_cities_2024.xlsx (ITA top cities visited)

Usage:
    python src/add_tourism_feature.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

STATIONS_PATH = PROCESSED / "stations.csv"
FIPS_CACHE = PROCESSED / "station_county_fips.csv"
OMB_PATH = RAW / "list1_2023.xlsx"
TOURISM_PATH = RAW / "2024-Top-States-and-Cities-Visited.xlsx"

# ── Manual overrides for 5 tourism entries whose names don't substring-match ──
# Tourism name → OMB area code
TOURISM_NAME_OVERRIDES = {
    "Washington (DC Metro Area), DC-MD-VA": 47764,        # Washington, DC-MD
    "Hilo (Big Island), HI MICRO": 25900,                 # Hilo-Kailua, HI
    "Edison-New Brunswick, NJ MD": 29484,                  # Lakewood-New Brunswick, NJ
    "Barnstable Town (Cape Cod), MA MSA": 12700,           # Barnstable Town, MA
    "Kapaa (Kauai), HI MICRO": 28180,                     # Kapaa, HI
}


# ── Load OMB delineation ──────────────────────────────────────────────────────

def load_omb_delineation() -> pd.DataFrame:
    """
    Parse the OMB CBSA delineation file into a county_fips → area mapping.
    Uses Metropolitan Division code when available (more granular), otherwise
    falls back to CBSA code.
    """
    print(f"Loading {OMB_PATH.name} …")
    omb = pd.read_excel(OMB_PATH, header=2)

    # Drop footnote rows (non-numeric CBSA codes)
    omb = omb[pd.to_numeric(omb["CBSA Code"], errors="coerce").notna()].copy()

    # Build 5-digit county FIPS
    omb["county_fips"] = (
        omb["FIPS State Code"].fillna(0).astype(int).astype(str).str.zfill(2)
        + omb["FIPS County Code"].fillna(0).astype(int).astype(str).str.zfill(3)
    )

    # Use MD code if available (more granular), otherwise CBSA code
    omb["area_code"] = omb["Metropolitan Division Code"].fillna(omb["CBSA Code"])
    omb["area_title"] = omb["Metropolitan Division Title"].fillna(omb["CBSA Title"])

    # One row per county — filter to rows with valid numeric area codes
    county_map = (
        omb[["county_fips", "area_code", "area_title"]]
        .dropna(subset=["county_fips", "area_code"])
        .drop_duplicates("county_fips")
    )
    county_map = county_map[pd.to_numeric(county_map["area_code"], errors="coerce").notna()]
    county_map["area_code"] = pd.to_numeric(county_map["area_code"]).astype(int)

    print(f"  {len(county_map)} counties mapped to {county_map['area_code'].nunique()} areas")
    return county_map


# ── Load tourism data ─────────────────────────────────────────────────────────

def load_tourism(county_map: pd.DataFrame, top_n: int = 50) -> dict:
    """
    Parse tourism xlsx → match each entry to an OMB area code → return
    {area_code: visitors_thousands} lookup.

    Only the top_n destinations by visitor count are included. Stations outside
    those areas will receive NaN (not 0) — rural stations simply lack enough
    tourism to drive rail demand, they aren't confirmed zeros.
    """
    print(f"Loading {TOURISM_PATH.name} …")
    raw = pd.read_excel(TOURISM_PATH, sheet_name="Cities Visited-95CL", header=None)
    df = raw.iloc[5:123, [1, 3]].copy()
    df.columns = ["msa_name", "visitors_k"]
    df["visitors_k"] = pd.to_numeric(df["visitors_k"], errors="coerce")
    df = df.dropna(subset=["msa_name", "visitors_k"]).reset_index(drop=True)

    # Restrict to top N destinations before matching
    df = df.nlargest(top_n, "visitors_k").reset_index(drop=True)
    print(f"  Using top {top_n} tourism areas (threshold: {df['visitors_k'].min():,.0f}k visitors)")

    # Build set of unique OMB area titles for matching
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

    # Match each tourism entry to an OMB area code
    area_visitors = {}
    unmatched = []

    for _, row in df.iterrows():
        tname = str(row["msa_name"])
        visitors = row["visitors_k"]

        # Check manual overrides first
        if tname in TOURISM_NAME_OVERRIDES:
            code = TOURISM_NAME_OVERRIDES[tname]
            if code is not None:
                area_visitors[code] = area_visitors.get(code, 0) + visitors
            continue

        # Substring match: primary city + state in OMB area title
        pcity = extract_primary_city(tname).lower()
        state = extract_state(tname)

        matched = False
        for i, otitle in enumerate(omb_title_list):
            if pcity in otitle.lower() and (not state or state in otitle):
                code = omb_code_list[i]
                area_visitors[code] = area_visitors.get(code, 0) + visitors
                matched = True
                break

        if not matched:
            unmatched.append(tname)

    if unmatched:
        print(f"  Warning: {len(unmatched)} tourism entries unmatched:")
        for u in unmatched:
            print(f"    {u}")

    matched_count = len(df) - len(unmatched)
    print(f"  {matched_count}/{len(df)} tourism entries matched to OMB areas")
    return area_visitors


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for path, desc in [(STATIONS_PATH, "stations.csv"),
                        (FIPS_CACHE, "station_county_fips.csv"),
                        (OMB_PATH, "OMB delineation file"),
                        (TOURISM_PATH, "tourism xlsx")]:
        if not path.exists():
            sys.exit(f"Error: {path} not found ({desc})")

    stations = pd.read_csv(STATIONS_PATH, low_memory=False)
    print(f"Loaded {len(stations)} stations")

    fips_df = pd.read_csv(FIPS_CACHE, dtype=str)
    fips_df["county_fips"] = fips_df["county_fips"].replace("None", pd.NA)
    has_fips = fips_df["county_fips"].notna().sum()
    print(f"Loaded county FIPS for {has_fips}/{len(fips_df)} stations")

    county_map = load_omb_delineation()
    area_visitors = load_tourism(county_map)

    # Join: station → county FIPS → OMB area code → tourism visitors
    merged = stations.merge(fips_df[["code", "county_fips"]], on="code", how="left")
    merged = merged.merge(
        county_map[["county_fips", "area_code"]],
        on="county_fips",
        how="left",
    )
    # NaN for stations outside top-50 MSAs — don't treat as confirmed zero
    merged["overseas_visitors_thousands"] = (
        merged["area_code"]
        .map(area_visitors)
    )

    # Drop intermediate columns
    if "overseas_visitors_thousands" in stations.columns:
        stations = stations.drop(columns=["overseas_visitors_thousands"])

    # Replace the column in original stations (don't keep county_fips/area_code)
    stations["overseas_visitors_thousands"] = merged["overseas_visitors_thousands"].values

    has_tourism = stations["overseas_visitors_thousands"].notna().sum()
    print(f"\n  {has_tourism}/{len(stations)} stations with tourism data (rest are NaN)")

    stations.to_csv(STATIONS_PATH, index=False)
    print(f"  Updated {STATIONS_PATH} ({len(stations.columns)} cols)")

    print(f"\nTop 20 stations:")
    cols = ["station_name", "code", "overseas_visitors_thousands"]
    print(stations.nlargest(20, "overseas_visitors_thousands")[cols].to_string(index=False))

    print(f"\nFeature summary:")
    print(stations["overseas_visitors_thousands"].describe().round(0))


if __name__ == "__main__":
    main()