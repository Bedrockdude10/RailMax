"""
build_map.py

Generate results/underservice_map.html — a self-contained Leaflet.js map
visualising Amtrak station underservice with route segments colored by
the average demand ratio of their endpoints.

Features:
  - Each consecutive station pair on a route is a separate polyline
  - Segment color = avg(demand_ratio_A, demand_ratio_B)
  - Segment thickness ∝ 1/weekly_trips (more trips = thicker = more visible)
  - Station dots colored by individual demand_ratio
  - Filter controls for stations AND segments independently
  - What-if slider for weekly departures on each station popup
  - Expansion candidate cities with predicted ridership

Uses ALL service patterns (not just Mon–Fri) so tri-weekly routes like
the Sunset Limited and Cardinal are included.  Trip counts are expressed
as trips/week (sum of days_per_week across trip patterns for each route).

Data sources:
  data/raw/GTFS/              — stop_times, trips, routes, stops, calendar, shapes
  data/processed/stations.csv — station metadata + coordinates
  results/metrics/oof_predictions_v1.csv — OOF predictions from EBM

Run:
    python src/build_map.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RATIO_SEVERE,
    RATIO_UNDERSERVED,
    RATIO_BELOW_MODEL,
    RATIO_WELL_MATCHED_MAX,
    RATIO_SUPPRESSED,
    SUPPRESSED_MAX_TRIPS,
    SHAPE_DOWNSAMPLE,
    STATION_DISPLAY_GROUPS,
    TOP20_MIN_RIDERSHIP,
)
from utils import (
    load_amtrak_rail_gtfs,
    ratio_label,
    log_radius,
    dedup_segments_by_pair,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
OUT = ROOT / "results" / "underservice_map.html"
TEMPLATE_PATH = Path(__file__).resolve().parent / "map_template.html"
SHAPE_FUNCTIONS_CSV = METRICS / "shape_functions.csv"
EXPANSION_PREDICTIONS_CSV = METRICS / "expansion_predictions.csv"


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_weekly_departures_shape() -> dict:
    """
    Load the weekly_departures shape function bins and scores.
    Returns {"bins": [x0, x1, ...], "scores": [s0, s1, ...]} sorted by x.
    Used in the browser to compute what-if ridership estimates.
    """
    sf = pd.read_csv(SHAPE_FUNCTIONS_CSV)
    wd = sf[sf["feature"] == "weekly_departures"].copy()
    wd["x"] = pd.to_numeric(wd["x"], errors="coerce")
    wd = wd.dropna(subset=["x"]).sort_values("x")
    return {
        "bins":   wd["x"].round(2).tolist(),
        "scores": wd["score"].tolist(),
    }


def load_station_predictions() -> tuple[dict, pd.DataFrame]:
    """
    Merge OOF predictions with station coordinates.

    Returns:
        station_lookup  dict[code → {name, actual, predicted, ratio, lat, lon}]
        merged          DataFrame with all station data for building markers
    """
    print("Loading station predictions …")
    oof = pd.read_csv(METRICS / "oof_predictions_v1.csv")
    stations = pd.read_csv(PROCESSED / "stations.csv")

    oof["demand_ratio"] = (
        oof["actual_ridership"]
        / oof["oof_predicted_ridership"].replace(0, np.nan)
    ).round(3)

    merged = oof.merge(
        stations[["code", "station_name", "lat", "lon", "weekly_departures"]],
        on="code", how="left",
    )
    missing = merged["lat"].isna().sum()
    if missing:
        print(f"  Warning: {missing} stations without coordinates — skipping")
    merged = merged.dropna(subset=["lat", "lon"])
    print(f"  {len(merged)} stations with predictions + coordinates")

    # Build lookup by station code (used for segment scoring)
    station_lookup = {}
    for _, r in merged.iterrows():
        code = r["code"]
        if pd.isna(code):
            continue
        station_lookup[code] = {
            "name":      r["station_name"],
            "actual":    int(r["actual_ridership"]),
            "predicted": int(r["oof_predicted_ridership"]),
            "ratio":     float(r["demand_ratio"]) if pd.notna(r["demand_ratio"]) else None,
            "lat":       float(r["lat"]),
            "lon":       float(r["lon"]),
        }

    return station_lookup, merged


# ── Station marker records ────────────────────────────────────────────────────

def _make_station_record(name, actual, predicted, ratio, lat, lon, radius, wd):
    """Build a single station record dict for JSON serialisation."""
    color, label = ratio_label(ratio)
    return {
        "name":               name,
        "actual":             actual,
        "predicted":          predicted,
        "ratio":              ratio,
        "label":              label,
        "color":              color,
        "radius":             radius,
        "lat":                round(lat, 5),
        "lon":                round(lon, 5),
        "weekly_departures":  round(float(wd), 1) if wd is not None else None,
    }


def build_station_records(merged: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """
    Build JSON-ready station marker records from the merged predictions DataFrame.
    Handles display groups (co-located stations rendered as one dot).

    Returns:
        records     list[dict] for JSON
        top20       list[str] top 20 underserved station names
    """
    radii = log_radius(merged["actual_ridership"].values)
    merged = merged.copy()
    merged["radius"] = radii

    # Track which codes are absorbed into a display group
    grouped_codes: set[str] = set()
    for _, member_codes in STATION_DISPLAY_GROUPS:
        grouped_codes.update(member_codes)

    records = []

    # Emit grouped dots first
    for display_name, member_codes in STATION_DISPLAY_GROUPS:
        members = merged[merged["code"].isin(member_codes)]
        if members.empty:
            continue
        actual_total = int(members["actual_ridership"].sum())
        predicted_total = int(members["oof_predicted_ridership"].sum())
        ratio = round(actual_total / predicted_total, 3) if predicted_total else None
        lat = float(members["lat"].mean())
        lon = float(members["lon"].mean())
        radius = float(log_radius(np.array([actual_total]))[0])
        wd = members["weekly_departures"].dropna()
        records.append(_make_station_record(
            display_name, actual_total, predicted_total, ratio,
            lat, lon, radius, float(wd.mean()) if len(wd) else None,
        ))

    # Emit all remaining (ungrouped) stations
    for _, r in merged.iterrows():
        if r["code"] in grouped_codes:
            continue
        ratio = r["demand_ratio"] if pd.notna(r["demand_ratio"]) else None
        wd = r["weekly_departures"] if pd.notna(r["weekly_departures"]) else None
        records.append(_make_station_record(
            r["station_name"], int(r["actual_ridership"]),
            int(r["oof_predicted_ridership"]),
            float(ratio) if ratio is not None else None,
            float(r["lat"]), float(r["lon"]), float(r["radius"]), wd,
        ))

    # Top 20 underserved: lowest ratio, min ridership threshold
    top20 = list(dict.fromkeys(
        merged[merged["actual_ridership"] > TOP20_MIN_RIDERSHIP]
        .nsmallest(20, "demand_ratio")["station_name"]
        .tolist()
    ))
    print(f"  Top 20 underserved: {top20[:5]} …")

    return records, top20


# ── Expansion candidates ──────────────────────────────────────────────────────

def load_expansion_candidates() -> list[dict]:
    """
    Load predicted ridership for expansion candidate cities.
    Returns list[dict] ready for JSON, or empty list if file missing.
    """
    if not EXPANSION_PREDICTIONS_CSV.exists():
        print("  No expansion_predictions.csv found — skipping")
        return []

    df = pd.read_csv(EXPANSION_PREDICTIONS_CSV)
    candidates_csv = PROCESSED / "expansion_candidates.csv"
    if candidates_csv.exists():
        cands = pd.read_csv(candidates_csv, low_memory=False)
        df = df.merge(
            cands[["City", "lat", "lon"]].rename(columns={"City": "city"}),
            on="city", how="left",
        )
    else:
        df["lat"] = np.nan
        df["lon"] = np.nan

    df = df.dropna(subset=["lat", "lon"])
    print(f"  {len(df)} expansion candidates with coordinates")

    radii = log_radius(df["predicted_annual_ridership"].clip(lower=0).values)

    records = []
    for i, (_, r) in enumerate(df.iterrows()):
        records.append({
            "name":       str(r["city"]),
            "predicted":  int(r["predicted_annual_ridership"]),
            "lat":        round(float(r["lat"]), 5),
            "lon":        round(float(r["lon"]), 5),
            "radius":     float(radii[i]),
            "population": int(r["population"]) if pd.notna(r.get("population")) else None,
            "has_intercity_bus": int(r["has_intercity_bus"]) if pd.notna(r.get("has_intercity_bus")) else 0,
        })

    return records


# ── Route segments ────────────────────────────────────────────────────────────

def _snap_stops_to_shape(stop_ids, station_lookup, shape_lats, shape_lons):
    """
    Map each stop to the nearest point on the GTFS shape polyline,
    enforcing monotonically increasing indices along the shape.
    """
    indices = []
    for sid in stop_ids:
        s = station_lookup.get(sid)
        if not s:
            indices.append(None)
            continue
        dists = (shape_lats - s["lat"]) ** 2 + (shape_lons - s["lon"]) ** 2
        floor = indices[-1] if indices and indices[-1] is not None else 0
        valid = np.arange(floor, len(dists))
        if len(valid) == 0:
            indices.append(len(dists) - 1)
        else:
            indices.append(int(valid[np.argmin(dists[valid])]))
    return indices


def _extract_segment_coords(shape_lats, shape_lons, idx_a, idx_b, a, b):
    """
    Extract and downsample shape coordinates for one segment,
    or fall back to a straight line between endpoints.
    """
    if (shape_lats is not None and idx_a is not None
            and idx_b is not None and idx_b > idx_a):
        seg_lats = shape_lats[idx_a : idx_b + 1]
        seg_lons = shape_lons[idx_a : idx_b + 1]
        # Downsample, always keeping first and last point
        indices = list(range(0, len(seg_lats), SHAPE_DOWNSAMPLE))
        if (len(seg_lats) - 1) not in indices:
            indices.append(len(seg_lats) - 1)
        return [
            [round(float(seg_lats[j]), 4), round(float(seg_lons[j]), 4)]
            for j in indices
        ]

    # Straight line fallback
    return [
        [round(a["lat"], 4), round(a["lon"], 4)],
        [round(b["lat"], 4), round(b["lon"], 4)],
    ]


def _segment_score(a_ratio, b_ratio):
    """Average of endpoint demand ratios, or whichever is available."""
    if a_ratio is not None and b_ratio is not None:
        return round((a_ratio + b_ratio) / 2, 3)
    return a_ratio if a_ratio is not None else b_ratio


def build_segments(gtfs, station_lookup):
    """
    For each route, extract the representative trip's ordered station sequence,
    split the GTFS shape geometry at each stop, and compute segment-level
    underservice scores.

    Returns list[dict] ready for JSON serialisation.
    """
    print("\nBuilding route segments …")
    rail = gtfs["routes"]
    rail_trips = gtfs["trips"]
    stop_times = gtfs["stop_times"]
    stops = gtfs["stops"]
    shapes = gtfs["shapes"]

    # Extend station_lookup with GTFS stops not in our station set
    for _, s in stops.iterrows():
        if s["stop_id"] not in station_lookup:
            station_lookup[s["stop_id"]] = {
                "name": s["stop_name"],
                "actual": None, "predicted": None, "ratio": None,
                "lat": float(s["stop_lat"]),
                "lon": float(s["stop_lon"]),
            }

    segments = []

    for _, r in rail.iterrows():
        route_id = r["route_id"]
        route_name = r["route_long_name"]

        route_trips = rail_trips[rail_trips["route_id"] == route_id]
        if len(route_trips) == 0:
            continue

        weekly_trips = int(route_trips["days_per_week"].sum())

        # Representative trip: the one with the most stops
        trip_sizes = (
            stop_times[stop_times["trip_id"].isin(route_trips["trip_id"])]
            .groupby("trip_id").size()
        )
        best_trip = trip_sizes.idxmax()
        trip_stops = (
            stop_times[stop_times["trip_id"] == best_trip]
            .sort_values("stop_sequence")
        )
        stop_ids = trip_stops["stop_id"].tolist()

        # Get shape geometry for this trip (if available)
        trip_row = rail_trips[rail_trips["trip_id"] == best_trip].iloc[0]
        shape_id = trip_row.get("shape_id")
        has_shape = pd.notna(shape_id) if shape_id is not None else False

        if has_shape:
            shape_pts = (
                shapes[shapes["shape_id"] == shape_id]
                .sort_values("shape_pt_sequence")
            )
            shape_lats = shape_pts["shape_pt_lat"].values
            shape_lons = shape_pts["shape_pt_lon"].values
            stop_shape_idx = _snap_stops_to_shape(
                stop_ids, station_lookup, shape_lats, shape_lons,
            )
        else:
            shape_lats = shape_lons = None
            stop_shape_idx = [None] * len(stop_ids)

        # Build segment for each consecutive pair
        for i in range(len(stop_ids) - 1):
            a_code, b_code = stop_ids[i], stop_ids[i + 1]
            a = station_lookup.get(a_code, {})
            b = station_lookup.get(b_code, {})
            if not a or not b:
                continue

            a_ratio = a.get("ratio")
            b_ratio = b.get("ratio")

            segments.append({
                "route":    route_name,
                "from":     a.get("name", a_code),
                "to":       b.get("name", b_code),
                "coords":   _extract_segment_coords(
                    shape_lats, shape_lons,
                    stop_shape_idx[i], stop_shape_idx[i + 1], a, b,
                ),
                "score":    _segment_score(a_ratio, b_ratio),
                "trips":    weekly_trips,
                "from_r":   a_ratio,
                "to_r":     b_ratio,
                "from_rid": a.get("actual"),
                "to_rid":   b.get("actual"),
            })

    total_pts = sum(len(s["coords"]) for s in segments)
    scored = sum(1 for s in segments if s["score"] is not None)
    routes_repr = len({s["route"] for s in segments})
    print(f"  {len(segments)} segments across {routes_repr} routes")
    print(f"  {scored} segments with underservice scores")
    print(f"  {total_pts:,} shape points after downsampling")

    return segments


# ── Station min-trips enrichment ──────────────────────────────────────────────

def enrich_stations_with_min_trips(station_records, segments):
    """Add min_trips (lowest weekly trips across all segments touching a station)."""
    station_min_trips: dict[str, int] = {}
    for seg in segments:
        for name in (seg["from"], seg["to"]):
            trips = seg["trips"]
            if name not in station_min_trips or trips < station_min_trips[name]:
                station_min_trips[name] = trips
    for rec in station_records:
        rec["min_trips"] = station_min_trips.get(rec["name"])


# ── HTML assembly ─────────────────────────────────────────────────────────────

def render_html(segments, station_records, top20, wd_shape, expansion_records) -> str:
    """Load the HTML template and inject all JSON data and config thresholds."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    thresholds = {
        "severe":             RATIO_SEVERE,
        "underserved":        RATIO_UNDERSERVED,
        "belowModel":         RATIO_BELOW_MODEL,
        "wellMatched":        RATIO_WELL_MATCHED_MAX,
        "suppressed":         RATIO_SUPPRESSED,
        "suppressedMaxTrips": SUPPRESSED_MAX_TRIPS,
    }

    replacements = {
        "__SEGMENTS_JSON__":        json.dumps(segments, separators=(",", ":")),
        "__STATIONS_JSON__":        json.dumps(station_records, separators=(",", ":")),
        "__TOP20_JSON__":           json.dumps(top20, separators=(",", ":")),
        "__WD_SHAPE_JSON__":        json.dumps(wd_shape, separators=(",", ":")),
        "__CANDIDATES_JSON__":      json.dumps(expansion_records, separators=(",", ":")),
        "__THRESHOLDS_JSON__":      json.dumps(thresholds, separators=(",", ":")),
        "__SUPPRESSED_MAX_TRIPS__": str(SUPPRESSED_MAX_TRIPS),
    }
    for placeholder, data in replacements.items():
        template = template.replace(placeholder, data)

    return template


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_underserved_summary(segments):
    """Print the top underserved segments and low-frequency underserved segments."""
    scored = sorted(
        [s for s in segments if s["score"] is not None],
        key=lambda s: s["score"],
    )

    print("\nTop 10 underserved segments (by avg endpoint ratio):")
    for s in dedup_segments_by_pair(scored, 10):
        print(f"  {s['route']:30s} {s['from']:>30s} → {s['to']:<30s} "
              f"score={s['score']:.2f}  trips/wk={s['trips']}")

    low_freq = [s for s in scored if s["trips"] <= SUPPRESSED_MAX_TRIPS]
    print(f"\nTop 10 underserved + low-frequency (≤{SUPPRESSED_MAX_TRIPS}/wk) segments:")
    for s in dedup_segments_by_pair(low_freq, 10):
        print(f"  {s['route']:30s} {s['from']:>30s} → {s['to']:<30s} "
              f"score={s['score']:.2f}  trips/wk={s['trips']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load data
    gtfs = load_amtrak_rail_gtfs(include_shapes=True, exclude_auto_train=True)
    station_lookup, merged = load_station_predictions()
    station_records, top20 = build_station_records(merged)

    # Build segments and enrich stations with min trip counts
    segments = build_segments(gtfs, station_lookup)
    enrich_stations_with_min_trips(station_records, segments)

    # Load ancillary data
    print("\nLoading expansion candidates …")
    expansion_records = load_expansion_candidates()
    wd_shape = load_weekly_departures_shape()
    print(f"  Loaded weekly_departures shape: {len(wd_shape['bins'])} bins "
          f"(range {wd_shape['bins'][0]}–{wd_shape['bins'][-1]} trips/wk)")

    # Render and write HTML
    html = render_html(segments, station_records, top20, wd_shape, expansion_records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")

    size_kb = OUT.stat().st_size / 1024
    print(f"\nWrote {OUT}  ({size_kb:.0f} KB)")
    assert size_kb < 2048, f"Output is {size_kb:.0f} KB — exceeds 2MB budget!"

    print_underserved_summary(segments)


if __name__ == "__main__":
    main()