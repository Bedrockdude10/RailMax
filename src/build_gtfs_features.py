"""
build_gtfs_features.py

Compute 11 GTFS-derived features per Amtrak station and join them to
data/processed/stations.csv, writing the result back in-place.

Filters applied:
  - route_type = 2 (rail only, no Thruway buses)
  - agency_id = 51 (Amtrak only)

"Typical weekday" = service_id where monday=1 AND friday=1
(proxy for a regular Mon-Fri service pattern).

Run directly:
    python src/build_gtfs_features.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import process as fuzz_process, fuzz

ROOT = Path(__file__).resolve().parent.parent
GTFS = ROOT / "data" / "raw" / "GTFS"
PROCESSED = ROOT / "data" / "processed"


# ── Helpers ────────────────────────────────────────────────────────────────────

def time_to_sec(t: str) -> float:
    """Convert 'HH:MM:SS' (may exceed 24h) to total seconds."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two lat/lon pairs."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ── Load & filter GTFS ────────────────────────────────────────────────────────

def load_gtfs():
    print("Loading GTFS files …")
    agency   = pd.read_csv(GTFS / "agency.txt")
    routes   = pd.read_csv(GTFS / "routes.txt")
    calendar = pd.read_csv(GTFS / "calendar.txt")
    trips    = pd.read_csv(GTFS / "trips.txt")
    stop_times = pd.read_csv(GTFS / "stop_times.txt")
    stops    = pd.read_csv(GTFS / "stops.txt")

    # Rail-only, Amtrak-only routes
    rail_routes = routes[
        (routes["route_type"] == 2) & (routes["agency_id"] == 51)
    ]["route_id"].unique()
    print(f"  Rail routes (agency 51): {len(rail_routes)}")

    # Weekday service_ids: monday=1 AND friday=1
    weekday_sids = calendar[
        (calendar["monday"] == 1) & (calendar["friday"] == 1)
    ]["service_id"].unique()
    print(f"  Weekday service_ids: {len(weekday_sids)}")

    # Trips: weekday + rail routes
    trips_filt = trips[
        trips["route_id"].isin(rail_routes) &
        trips["service_id"].isin(weekday_sids)
    ].copy()
    print(f"  Weekday rail trips: {len(trips_filt)}")

    # Keep only stop_times for filtered trips
    st_filt = stop_times[stop_times["trip_id"].isin(trips_filt["trip_id"])].copy()
    print(f"  Stop-time rows (weekday rail): {len(st_filt)}")

    return rail_routes, trips_filt, st_filt, stops


# ── Per-stop features ─────────────────────────────────────────────────────────

def compute_per_stop_features(rail_routes, trips_filt, st_filt, stops):
    print("\nComputing per-stop features …")

    # Merge trip metadata into stop_times
    st = st_filt.merge(
        trips_filt[["trip_id", "route_id", "direction_id"]],
        on="trip_id", how="left"
    )

    # Convert times to seconds (handles >24h overnight times)
    st["arr_sec"] = st["arrival_time"].apply(time_to_sec)
    st["dep_sec"] = st["departure_time"].apply(time_to_sec)
    st["dwell_sec"] = st["dep_sec"] - st["arr_sec"]

    # Max stop_sequence per trip (for terminal detection and pct calc)
    trip_max_seq = st.groupby("trip_id")["stop_sequence"].max().rename("max_seq")
    st = st.merge(trip_max_seq, on="trip_id")
    st["seq_pct"] = st["stop_sequence"] / st["max_seq"]

    # Terminal: first or last stop on any trip
    st["is_first"] = (st["stop_sequence"] == 1).astype(int)
    st["is_last"]  = (st["stop_sequence"] == st["max_seq"]).astype(int)

    # ── Route lengths: Haversine(first stop → last stop) ──
    # Get first and last stop coordinates for each trip
    first_stops = (
        st[st["is_first"] == 1][["trip_id", "stop_id"]]
        .merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id")
        .rename(columns={"stop_lat": "lat1", "stop_lon": "lon1"})
    )
    last_stops = (
        st[st["is_last"] == 1][["trip_id", "stop_id"]]
        .merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id")
        .rename(columns={"stop_lat": "lat2", "stop_lon": "lon2"})
    )
    trip_lengths = first_stops.merge(last_stops, on="trip_id", suffixes=("_f", "_l"))
    trip_lengths["route_len_km"] = trip_lengths.apply(
        lambda r: haversine_km(r["lat1"], r["lon1"], r["lat2"], r["lon2"]), axis=1
    )

    # Map trip → route, then route → max trip length (representative length)
    trip_route = trips_filt[["trip_id", "route_id"]].drop_duplicates()
    trip_lengths = trip_lengths.merge(trip_route, on="trip_id")
    route_len = (
        trip_lengths.groupby("route_id")["route_len_km"]
        .max()  # use max trip length as route length
        .reset_index()
    )

    # Attach route length to stop_times
    st = st.merge(route_len, on="route_id", how="left")

    # ── Aggregate per stop_id ──
    agg = {}

    # daily_departures / num_weekday_trips: distinct trip_ids per stop
    daily_dep = st.groupby("stop_id")["trip_id"].nunique().rename("daily_departures")
    agg["daily_departures"] = daily_dep

    # num_routes_served
    num_routes = st.groupby("stop_id")["route_id"].nunique().rename("num_routes_served")
    agg["num_routes_served"] = num_routes

    # is_terminal
    is_term = st.groupby("stop_id").apply(
        lambda g: int((g["is_first"].any()) or (g["is_last"].any())),
        include_groups=False,
    ).rename("is_terminal")
    agg["is_terminal"] = is_term

    # avg_dwell_time_sec
    avg_dwell = st.groupby("stop_id")["dwell_sec"].mean().rename("avg_dwell_time_sec")
    agg["avg_dwell_time_sec"] = avg_dwell

    # service_span_hours: (max dep - min dep) on weekday
    span = st.groupby("stop_id").apply(
        lambda g: (g["dep_sec"].max() - g["dep_sec"].min()) / 3600.0,
        include_groups=False,
    ).rename("service_span_hours")
    agg["service_span_hours"] = span

    # avg_stop_sequence_pct
    avg_seq_pct = st.groupby("stop_id")["seq_pct"].mean().rename("avg_stop_sequence_pct")
    agg["avg_stop_sequence_pct"] = avg_seq_pct

    # num_directions
    num_dir = st.groupby("stop_id")["direction_id"].nunique().rename("num_directions")
    agg["num_directions"] = num_dir

    # avg_route_length_km and max_route_length_km
    avg_rlen = st.groupby("stop_id")["route_len_km"].mean().rename("avg_route_length_km")
    max_rlen = st.groupby("stop_id")["route_len_km"].max().rename("max_route_length_km")
    agg["avg_route_length_km"] = avg_rlen
    agg["max_route_length_km"] = max_rlen

    # num_weekday_trips (same as daily_departures, kept as separate column per spec)
    agg["num_weekday_trips"] = daily_dep.rename("num_weekday_trips")

    # pct_long_distance: fraction of routes > 500km serving this stop
    def pct_long(g):
        route_lens = g.groupby("route_id")["route_len_km"].max()
        return (route_lens > 500).mean()

    pct_ld = st.groupby("stop_id").apply(pct_long, include_groups=False).rename("pct_long_distance")
    agg["pct_long_distance"] = pct_ld

    # Combine
    feat_df = pd.concat(agg.values(), axis=1).reset_index()
    feat_df.columns = ["stop_id"] + list(agg.keys())

    # Attach stop coordinates and name for matching
    feat_df = feat_df.merge(
        stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
        on="stop_id", how="left"
    )

    print(f"  Features computed for {len(feat_df)} stops.")
    return feat_df


# ── Match GTFS stops → stations.csv ───────────────────────────────────────────

COORD_MATCH_KM = 2.0  # max distance for coordinate fallback


def match_stops_to_stations(feat_df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """
    For each GTFS stop, find the best matching station in stations.csv.
    Strategy:
      1. Exact stop_id match on stations 'code' column.
      2. Fuzzy name match (rapidfuzz, score >= 85).
      3. Nearest coordinate within COORD_MATCH_KM.
    Returns feat_df with a new 'matched_code' column.
    """
    print("\nMatching GTFS stops to stations.csv …")

    station_codes  = stations["code"].tolist()
    station_names  = stations["station_name"].tolist()
    station_lats   = stations["lat"].tolist()
    station_lons   = stations["lon"].tolist()

    matched_codes = []
    match_methods = []

    for _, row in feat_df.iterrows():
        sid   = row["stop_id"]
        sname = row["stop_name"]
        slat  = row["stop_lat"]
        slon  = row["stop_lon"]

        code = None
        method = None

        # 1. Exact station code
        if sid in station_codes:
            code = sid
            method = "exact_code"

        # 2. Fuzzy name match
        if code is None:
            # Strip " Amtrak Station" etc. for cleaner match
            clean_name = (sname
                          .replace(" Amtrak Station", "")
                          .replace(" Station", "")
                          .replace(" Train Station", "")
                          .strip())
            # Compare against station_name column (e.g. "Aberdeen, MD")
            # Also try the City part only
            result = fuzz_process.extractOne(
                clean_name,
                station_names,
                scorer=fuzz.token_set_ratio,
                score_cutoff=80,
            )
            if result is not None:
                matched_name, score, idx = result
                code = station_codes[idx]
                method = f"fuzzy_name(score={score:.0f})"

        # 3. Coordinate proximity
        if code is None and not (math.isnan(slat) or math.isnan(slon)):
            min_dist = float("inf")
            best_idx = -1
            for i, (lat2, lon2) in enumerate(zip(station_lats, station_lons)):
                if math.isnan(lat2) or math.isnan(lon2):
                    continue
                d = haversine_km(slat, slon, lat2, lon2)
                if d < min_dist:
                    min_dist = d
                    best_idx = i
            if min_dist <= COORD_MATCH_KM and best_idx >= 0:
                code = station_codes[best_idx]
                method = f"coord(dist={min_dist:.2f}km)"

        matched_codes.append(code)
        match_methods.append(method or "unmatched")

    feat_df = feat_df.copy()
    feat_df["matched_code"] = matched_codes
    feat_df["match_method"] = match_methods

    matched = feat_df["matched_code"].notna().sum()
    print(f"  Matched {matched}/{len(feat_df)} GTFS stops to stations.")
    unmatched = feat_df[feat_df["matched_code"].isna()]["stop_name"].tolist()
    if unmatched:
        print(f"  Unmatched stops ({len(unmatched)}): {unmatched[:10]}")
    return feat_df


GTFS_FEATURE_COLS = [
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


def join_to_stations(feat_df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-station (some GTFS stops may map to the same station),
    then left-join onto stations.csv.
    """
    matched = feat_df[feat_df["matched_code"].notna()].copy()

    # If multiple GTFS stops map to the same station code, aggregate
    agg_rules = {col: "mean" for col in GTFS_FEATURE_COLS}
    # For integer/binary columns use max instead of mean where it makes sense
    agg_rules["is_terminal"] = "max"
    agg_rules["num_directions"] = "max"
    agg_rules["daily_departures"] = "sum"
    agg_rules["num_weekday_trips"] = "sum"
    agg_rules["num_routes_served"] = "sum"

    station_feats = matched.groupby("matched_code").agg(agg_rules).reset_index()
    station_feats = station_feats.rename(columns={"matched_code": "code"})

    # Drop old GTFS columns from stations if they exist (re-running script)
    for col in GTFS_FEATURE_COLS:
        if col in stations.columns:
            stations = stations.drop(columns=[col])

    merged = stations.merge(station_feats[["code"] + GTFS_FEATURE_COLS],
                            on="code", how="left")
    filled = merged[GTFS_FEATURE_COLS].notna().all(axis=1).sum()
    print(f"\nJoined GTFS features to {filled}/{len(merged)} stations.")
    return merged


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    stations = pd.read_csv(PROCESSED / "stations.csv", low_memory=False)
    print(f"Loaded {len(stations)} stations from stations.csv")

    rail_routes, trips_filt, st_filt, stops = load_gtfs()
    feat_df = compute_per_stop_features(rail_routes, trips_filt, st_filt, stops)
    feat_df = match_stops_to_stations(feat_df, stations)

    # Diagnostics: show a sample of matches
    sample = feat_df[feat_df["matched_code"].notna()].head(10)[
        ["stop_id", "stop_name", "matched_code", "match_method",
         "daily_departures", "num_routes_served", "avg_route_length_km"]
    ]
    print("\nSample matches:")
    print(sample.to_string(index=False))

    merged = join_to_stations(feat_df, stations)

    out_path = PROCESSED / "stations.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nUpdated stations.csv written to {out_path}")
    print(f"Columns now: {merged.columns.tolist()}")

    # Summary stats on GTFS features
    print("\nGTFS feature summary (stations with data):")
    print(merged[GTFS_FEATURE_COLS].describe().round(2).to_string())


if __name__ == "__main__":
    # Check for rapidfuzz
    try:
        import rapidfuzz  # noqa: F401
    except ImportError:
        print("Installing rapidfuzz …")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rapidfuzz", "-q"])
        import rapidfuzz  # noqa: F401

    main()
