"""
build_gtfs_features.py

Compute GTFS-derived features per Amtrak station and join them to
data/processed/stations.csv, writing the result back in-place.

Filters applied:
  - route_type = 2 (rail only, no Thruway buses)
  - agency_id = 51 (Amtrak only)

Uses ALL service patterns (not just Mon–Fri) so tri-weekly routes like
the Sunset Limited and Cardinal are included.  Departure/trip counts
are expressed per week using the sum of days_per_week from calendar.txt.

Run directly:
    python src/build_gtfs_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import process as fuzz_process, fuzz

from config import GTFS_COORD_MATCH_KM, GTFS_FUZZY_SCORE, LONG_DISTANCE_ROUTE_KM
from utils import haversine_km

ROOT = Path(__file__).resolve().parent.parent
GTFS = ROOT / "data" / "raw" / "GTFS"
PROCESSED = ROOT / "data" / "processed"


# ── Helpers ────────────────────────────────────────────────────────────────────

def time_to_sec(t: str) -> float:
    """Convert 'HH:MM:SS' (may exceed 24h) to total seconds."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


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

    # All rail trips (no weekday filter — include tri-weekly routes etc.)
    trips_filt = trips[trips["route_id"].isin(rail_routes)].copy()
    print(f"  Rail trip patterns (all service days): {len(trips_filt)}")

    # Compute days_per_week per service_id for weekly trip counts
    calendar["days_per_week"] = (
        calendar["monday"] + calendar["tuesday"] + calendar["wednesday"]
        + calendar["thursday"] + calendar["friday"]
        + calendar["saturday"] + calendar["sunday"]
    )
    trips_filt = trips_filt.merge(
        calendar[["service_id", "days_per_week"]], on="service_id", how="left"
    )
    trips_filt["days_per_week"] = trips_filt["days_per_week"].fillna(0)

    # Keep only stop_times for filtered trips
    st_filt = stop_times[stop_times["trip_id"].isin(trips_filt["trip_id"])].copy()
    print(f"  Stop-time rows (all rail): {len(st_filt)}")

    return rail_routes, trips_filt, st_filt, stops


# ── Per-stop features ─────────────────────────────────────────────────────────

def compute_per_stop_features(rail_routes, trips_filt, st_filt, stops):
    print("\nComputing per-stop features …")

    # Merge trip metadata into stop_times
    st = st_filt.merge(
        trips_filt[["trip_id", "route_id", "direction_id", "days_per_week"]],
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
    trip_lengths["route_len_km"] = haversine_km(
        trip_lengths["lat1"].values, trip_lengths["lon1"].values,
        trip_lengths["lat2"].values, trip_lengths["lon2"].values,
    )

    # Map trip → route, then route → max trip length (representative length)
    trip_route = trips_filt[["trip_id", "route_id"]].drop_duplicates()
    trip_lengths = trip_lengths.merge(trip_route, on="trip_id")
    route_len = (
        trip_lengths.groupby("route_id")["route_len_km"]
        .max()
        .reset_index()
    )

    # Attach route length to stop_times
    st = st.merge(route_len, on="route_id", how="left")

    # ── Aggregate per stop_id ──
    agg = {}

    # weekly_departures: sum of days_per_week across trip patterns per stop
    agg["weekly_departures"] = st.groupby("stop_id")["days_per_week"].sum().rename("weekly_departures")

    # num_routes_served: count of distinct routes calling at this stop
    agg["num_routes_served"] = st.groupby("stop_id")["route_id"].nunique().rename("num_routes_served")

    # is_terminal
    agg["is_terminal"] = st.groupby("stop_id").apply(
        lambda g: int((g["is_first"].any()) or (g["is_last"].any())),
        include_groups=False,
    ).rename("is_terminal")

    # avg_dwell_time_sec
    agg["avg_dwell_time_sec"] = st.groupby("stop_id")["dwell_sec"].mean().rename("avg_dwell_time_sec")

    # service_span_hours: (max dep - min dep) across all service patterns
    agg["service_span_hours"] = st.groupby("stop_id").apply(
        lambda g: (g["dep_sec"].max() - g["dep_sec"].min()) / 3600.0,
        include_groups=False,
    ).rename("service_span_hours")

    # avg_stop_sequence_pct
    agg["avg_stop_sequence_pct"] = st.groupby("stop_id")["seq_pct"].mean().rename("avg_stop_sequence_pct")

    # num_directions
    agg["num_directions"] = st.groupby("stop_id")["direction_id"].nunique().rename("num_directions")

    # avg_route_length_km and max_route_length_km
    agg["avg_route_length_km"] = st.groupby("stop_id")["route_len_km"].mean().rename("avg_route_length_km")
    agg["max_route_length_km"] = st.groupby("stop_id")["route_len_km"].max().rename("max_route_length_km")

    # pct_long_distance: fraction of routes > LONG_DISTANCE_ROUTE_KM serving this stop
    def pct_long(g):
        route_lens = g.groupby("route_id")["route_len_km"].max()
        return (route_lens > LONG_DISTANCE_ROUTE_KM).mean()

    agg["pct_long_distance"] = st.groupby("stop_id").apply(pct_long, include_groups=False).rename("pct_long_distance")

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

def match_stops_to_stations(feat_df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """
    For each GTFS stop, find the best matching station in stations.csv.
    Strategy:
      1. Exact stop_id match on stations 'code' column.
      2. Fuzzy name match (rapidfuzz, score >= GTFS_FUZZY_SCORE).
      3. Nearest coordinate within GTFS_COORD_MATCH_KM (vectorised).
    Returns feat_df with a new 'matched_code' column.
    """
    print("\nMatching GTFS stops to stations.csv …")

    station_codes  = stations["code"].values
    station_names  = stations["station_name"].tolist()
    station_lats   = stations["lat"].values
    station_lons   = stations["lon"].values

    matched_codes = []
    match_methods = []

    # Pre-filter stations with valid coordinates for coord fallback
    valid_station_mask = ~(np.isnan(station_lats) | np.isnan(station_lons))
    valid_station_codes = station_codes[valid_station_mask]
    valid_s_lats = station_lats[valid_station_mask]
    valid_s_lons = station_lons[valid_station_mask]

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
            clean_name = (sname
                          .replace(" Amtrak Station", "")
                          .replace(" Station", "")
                          .replace(" Train Station", "")
                          .strip())
            result = fuzz_process.extractOne(
                clean_name,
                station_names,
                scorer=fuzz.token_set_ratio,
                score_cutoff=GTFS_FUZZY_SCORE,
            )
            if result is not None:
                matched_name, score, idx = result
                code = station_codes[idx]
                method = f"fuzzy_name(score={score:.0f})"

        # 3. Coordinate proximity (vectorised)
        if code is None and not (np.isnan(slat) or np.isnan(slon)):
            dists = haversine_km(slat, slon, valid_s_lats, valid_s_lons)
            min_idx = np.argmin(dists)
            if dists[min_idx] <= GTFS_COORD_MATCH_KM:
                code = valid_station_codes[min_idx]
                method = f"coord(dist={dists[min_idx]:.2f}km)"

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
    "weekly_departures",
    "num_routes_served",
    "is_terminal",
    "avg_dwell_time_sec",
    "service_span_hours",
    "avg_stop_sequence_pct",
    "num_directions",
    "avg_route_length_km",
    "max_route_length_km",
    "pct_long_distance",
]


def join_to_stations(feat_df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-station (some GTFS stops may map to the same station),
    then left-join onto stations.csv.
    """
    matched = feat_df[feat_df["matched_code"].notna()].copy()

    agg_rules = {col: "mean" for col in GTFS_FEATURE_COLS}
    agg_rules["is_terminal"] = "max"
    agg_rules["num_directions"] = "max"
    agg_rules["weekly_departures"] = "sum"
    # num_routes_served: when multiple GTFS stops map to the same station,
    # sum the per-stop unique-route counts as an upper-bound approximation.
    # (True unique-route count across stops would need a second-pass groupby.)
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
         "weekly_departures", "num_routes_served", "avg_route_length_km"]
    ]
    print("\nSample matches:")
    print(sample.to_string(index=False))

    merged = join_to_stations(feat_df, stations)

    out_path = PROCESSED / "stations.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nUpdated stations.csv written to {out_path}")
    print(f"Columns now: {merged.columns.tolist()}")

    print("\nGTFS feature summary (stations with data):")
    print(merged[GTFS_FEATURE_COLS].describe().round(2).to_string())


if __name__ == "__main__":
    try:
        import rapidfuzz  # noqa: F401
    except ImportError:
        print("Installing rapidfuzz …")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rapidfuzz", "-q"])
        import rapidfuzz  # noqa: F401

    main()
