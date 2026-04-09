"""
utils.py

Shared utilities used across pipeline stages.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RATIO_SEVERE,
    RATIO_UNDERSERVED,
    RATIO_BELOW_MODEL,
    RATIO_WELL_MATCHED_MAX,
    MARKER_RADIUS_MIN,
    MARKER_RADIUS_MAX,
)


EARTH_RADIUS_KM = 6371.0

ROOT = Path(__file__).resolve().parent.parent
GTFS_DIR = ROOT / "data" / "raw" / "GTFS"


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorised Haversine distance (km).

    All arguments may be scalars or numpy arrays; broadcasting applies.
    Returns the same shape as the broadcast of the inputs.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [
        np.asarray(lat1, dtype=float),
        np.asarray(lon1, dtype=float),
        np.asarray(lat2, dtype=float),
        np.asarray(lon2, dtype=float),
    ])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Demand ratio display ──────────────────────────────────────────────────────

def ratio_label(ratio: float | None) -> tuple[str, str]:
    """
    Map a demand ratio (actual / predicted) to a (hex_color, label) pair.
    Used by build_map.py for station dots and popups.
    """
    if ratio is None:
        return "#888888", "unknown"
    if ratio < RATIO_SEVERE:
        return "#ef4444", f"{1/ratio:.1f}× underserved"
    if ratio < RATIO_UNDERSERVED:
        return "#f97316", f"{1/ratio:.1f}× underserved"
    if ratio < RATIO_BELOW_MODEL:
        return "#eab308", f"{1/ratio:.1f}× below model"
    if ratio <= RATIO_WELL_MATCHED_MAX:
        return "#22c55e", "well-matched"
    return "#06b6d4", f"{ratio:.1f}× outperforms model"


# ── Log-scaled marker radius ──────────────────────────────────────────────────

def log_radius(values: np.ndarray) -> np.ndarray:
    """
    Scale an array of positive values to marker radii using log1p.
    Returns radii in [MARKER_RADIUS_MIN, MARKER_RADIUS_MAX].
    """
    log_vals = np.log1p(values)
    lo, hi = log_vals.min(), log_vals.max()
    span = MARKER_RADIUS_MAX - MARKER_RADIUS_MIN
    if hi > lo:
        return np.round(MARKER_RADIUS_MIN + (log_vals - lo) / (hi - lo) * span, 1)
    return np.full_like(log_vals, MARKER_RADIUS_MIN + span / 2)


# ── GTFS loading (shared by build_gtfs_features.py and build_map.py) ──────────

def load_amtrak_rail_gtfs(
    include_shapes: bool = False,
    exclude_auto_train: bool = False,
) -> dict:
    """
    Load and filter GTFS to Amtrak rail service (all service patterns).

    Returns a dict with keys:
        routes, trips, stop_times, stops, calendar
        shapes (only if include_shapes=True)

    Trips are augmented with a `days_per_week` column computed from calendar.txt.
    """
    print("Loading GTFS files …")
    routes     = pd.read_csv(GTFS_DIR / "routes.txt")
    trips      = pd.read_csv(GTFS_DIR / "trips.txt")
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt")
    stops      = pd.read_csv(GTFS_DIR / "stops.txt")
    calendar   = pd.read_csv(GTFS_DIR / "calendar.txt")

    # Rail-only, Amtrak-only
    mask = (routes["route_type"] == 2) & (routes["agency_id"] == 51)
    if exclude_auto_train:
        mask = mask & (routes["route_long_name"] != "Auto Train")
    rail_routes = routes[mask]
    print(f"  {len(rail_routes)} rail routes")

    # All rail trips (no weekday filter — include tri-weekly routes etc.)
    rail_trips = trips[trips["route_id"].isin(rail_routes["route_id"])].copy()
    print(f"  {len(rail_trips)} rail trip patterns (all service days)")

    # Compute days_per_week per service_id
    day_cols = ["monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday"]
    calendar["days_per_week"] = calendar[day_cols].sum(axis=1)
    rail_trips = rail_trips.merge(
        calendar[["service_id", "days_per_week"]], on="service_id", how="left"
    )
    rail_trips["days_per_week"] = rail_trips["days_per_week"].fillna(0)

    # Filter stop_times to rail trips only
    rail_stop_times = stop_times[
        stop_times["trip_id"].isin(rail_trips["trip_id"])
    ].copy()
    print(f"  {len(rail_stop_times)} stop-time rows (all rail)")

    result = {
        "routes": rail_routes,
        "trips": rail_trips,
        "stop_times": rail_stop_times,
        "stops": stops,
        "calendar": calendar,
    }

    if include_shapes:
        result["shapes"] = pd.read_csv(GTFS_DIR / "shapes.txt")

    return result


# ── Segment deduplication ─────────────────────────────────────────────────────

def dedup_segments_by_pair(segments: list[dict], n: int) -> list[dict]:
    """
    Return the first `n` segments after deduplicating by unordered station pair.
    Segments should be pre-sorted by the desired ranking (e.g. ascending score).
    """
    seen: set[frozenset] = set()
    out = []
    for s in segments:
        key = frozenset([s["from"], s["to"]])
        if key not in seen:
            seen.add(key)
            out.append(s)
        if len(out) == n:
            break
    return out