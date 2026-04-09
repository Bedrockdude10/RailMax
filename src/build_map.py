"""
build_segment_map.py

Generate results/underservice_map.html — a self-contained Leaflet.js map
visualising Amtrak station underservice with route segments colored by
the average demand ratio of their endpoints.

Replaces build_map.py with segment-level detail:
  - Each consecutive station pair on a route is a separate polyline
  - Segment color = avg(demand_ratio_A, demand_ratio_B)
  - Segment thickness ∝ 1/weekly_trips (fewer trips = thicker = more visible)
  - Station dots unchanged (colored by individual demand_ratio)
  - Filter controls for stations AND segments independently

Uses ALL service patterns (not just Mon–Fri) so tri-weekly routes like
the Sunset Limited and Cardinal are included.  Trip counts are expressed
as trips/week (sum of days_per_week across trip patterns for each route).

Data sources:
  data/raw/GTFS/              — stop_times, trips, routes, stops, calendar, shapes
  data/processed/stations.csv — station metadata + coordinates
  results/metrics/oof_predictions_v1.csv — OOF predictions from EBM

Run:
    python src/build_segment_map.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
GTFS = RAW / "GTFS"
PROCESSED = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
OUT = ROOT / "results" / "underservice_map.html"

# Downsample shapes: keep every Nth point per segment.
SHAPE_DOWNSAMPLE = 6


# ── Load GTFS ─────────────────────────────────────────────────────────────────

def load_gtfs():
    """Load and filter GTFS to Amtrak rail service (all service patterns)."""
    print("Loading GTFS files …")
    routes   = pd.read_csv(GTFS / "routes.txt")
    trips    = pd.read_csv(GTFS / "trips.txt")
    stop_times = pd.read_csv(GTFS / "stop_times.txt")
    stops    = pd.read_csv(GTFS / "stops.txt")
    calendar = pd.read_csv(GTFS / "calendar.txt")
    shapes   = pd.read_csv(GTFS / "shapes.txt")

    # Rail-only, Amtrak-only, no Auto Train
    rail = routes[
        (routes["route_type"] == 2)
        & (routes["agency_id"] == 51)
        & (routes["route_long_name"] != "Auto Train")
    ]
    print(f"  {len(rail)} rail routes (excl. Auto Train)")

    # All rail trips (no weekday filter — include tri-weekly routes etc.)
    rail_trips = trips[trips["route_id"].isin(rail["route_id"])].copy()
    print(f"  {len(rail_trips)} rail trip patterns (all service days)")

    # Compute days_per_week per service_id for weekly trip counts
    calendar["days_per_week"] = (
        calendar["monday"] + calendar["tuesday"] + calendar["wednesday"]
        + calendar["thursday"] + calendar["friday"]
        + calendar["saturday"] + calendar["sunday"]
    )
    rail_trips = rail_trips.merge(
        calendar[["service_id", "days_per_week"]], on="service_id", how="left"
    )
    rail_trips["days_per_week"] = rail_trips["days_per_week"].fillna(0)

    return rail, rail_trips, stop_times, stops, shapes


# ── Load stations + OOF predictions ───────────────────────────────────────────

def load_stations():
    """
    Merge OOF predictions with station coordinates.
    Returns:
      station_lookup  dict[code → {name, actual, predicted, ratio, lat, lon}]
      station_records list[dict] for JSON (station markers)
      top20_names     list[str] top 20 underserved station names (actual > 50k)
    """
    print("Loading station predictions …")
    oof = pd.read_csv(METRICS / "oof_predictions_v1.csv")
    stations = pd.read_csv(PROCESSED / "stations.csv")

    oof["demand_ratio"] = (
        oof["actual_ridership"]
        / oof["oof_predicted_ridership"].replace(0, np.nan)
    ).round(3)

    merged = oof.merge(
        stations[["code", "station_name", "lat", "lon"]],
        on="code",
        how="left",
    )
    missing = merged["lat"].isna().sum()
    if missing:
        print(f"  Warning: {missing} stations without coordinates — skipping")
    merged = merged.dropna(subset=["lat", "lon"])

    print(f"  {len(merged)} stations with predictions + coordinates")

    # Build lookup by station code
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

    # Build station marker records for JSON
    log_rs = np.log1p(merged["actual_ridership"])
    lr_min, lr_max = log_rs.min(), log_rs.max()
    merged["radius"] = (5 + (log_rs - lr_min) / (lr_max - lr_min) * 17).round(1)

    records = []
    for _, r in merged.iterrows():
        ratio = r["demand_ratio"]
        if pd.isna(ratio):
            color, label = "#888888", "unknown"
        elif ratio < 0.33:
            # actual << predicted: strongly underserved
            color, label = "#ef4444", f"{1/ratio:.1f}× underserved"
        elif ratio < 0.5:
            color, label = "#f97316", f"{1/ratio:.1f}× underserved"
        elif ratio < 0.77:
            color, label = "#eab308", f"{1/ratio:.1f}× below model"
        elif ratio <= 1.3:
            color, label = "#22c55e", "well-matched"
        else:
            # actual >> predicted: outperforming model
            color, label = "#06b6d4", f"{ratio:.1f}× outperforms model"

        records.append({
            "name":      r["station_name"],
            "actual":    int(r["actual_ridership"]),
            "predicted": int(r["oof_predicted_ridership"]),
            "ratio":     float(ratio) if not pd.isna(ratio) else None,
            "label":     label,
            "color":     color,
            "radius":    float(r["radius"]),
            "lat":       round(float(r["lat"]), 5),
            "lon":       round(float(r["lon"]), 5),
        })

    # Top 20 underserved: predicted >> actual (lowest ratio), min 50k actual
    top20 = list(dict.fromkeys(
        merged[merged["actual_ridership"] > 50_000]
        .nsmallest(20, "demand_ratio")["station_name"]
        .tolist()
    ))
    print(f"  Top 20 underserved: {top20[:5]} …")

    return station_lookup, records, top20


# ── Build segments ────────────────────────────────────────────────────────────

def build_segments(rail, rail_trips, stop_times, stops, shapes, station_lookup):
    """
    For each route, extract the representative trip's ordered station sequence,
    split the GTFS shape geometry at each stop, and compute segment-level
    underservice scores.

    Returns list[dict] ready for JSON serialisation.
    """
    print("\nBuilding route segments …")

    # Extend station_lookup with GTFS stops not in our station set
    # (so we have coordinates for every stop_id)
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

            # Map each stop to the nearest shape point (monotonically increasing)
            stop_shape_idx = []
            for sid in stop_ids:
                s = station_lookup.get(sid)
                if not s:
                    stop_shape_idx.append(None)
                    continue
                dists = (shape_lats - s["lat"]) ** 2 + (shape_lons - s["lon"]) ** 2
                floor = stop_shape_idx[-1] if stop_shape_idx and stop_shape_idx[-1] is not None else 0
                valid = np.arange(floor, len(dists))
                if len(valid) == 0:
                    stop_shape_idx.append(len(dists) - 1)
                else:
                    stop_shape_idx.append(int(valid[np.argmin(dists[valid])]))
        else:
            stop_shape_idx = [None] * len(stop_ids)

        # Build segment for each consecutive pair
        for i in range(len(stop_ids) - 1):
            a_code, b_code = stop_ids[i], stop_ids[i + 1]
            a = station_lookup.get(a_code, {})
            b = station_lookup.get(b_code, {})
            if not a or not b:
                continue

            # Extract shape coords for this segment
            idx_a = stop_shape_idx[i]
            idx_b = stop_shape_idx[i + 1]

            if (has_shape and idx_a is not None and idx_b is not None
                    and idx_b > idx_a):
                seg_lats = shape_lats[idx_a : idx_b + 1]
                seg_lons = shape_lons[idx_a : idx_b + 1]
                # Downsample, always keeping first and last
                indices = list(range(0, len(seg_lats), SHAPE_DOWNSAMPLE))
                if (len(seg_lats) - 1) not in indices:
                    indices.append(len(seg_lats) - 1)
                coords = [
                    [round(float(seg_lats[j]), 4), round(float(seg_lons[j]), 4)]
                    for j in indices
                ]
            else:
                # Straight line fallback
                coords = [
                    [round(a["lat"], 4), round(a["lon"], 4)],
                    [round(b["lat"], 4), round(b["lon"], 4)],
                ]

            # Segment score = average of endpoint demand ratios
            a_ratio = a.get("ratio")
            b_ratio = b.get("ratio")
            if a_ratio is not None and b_ratio is not None:
                seg_score = round((a_ratio + b_ratio) / 2, 3)
            elif a_ratio is not None:
                seg_score = a_ratio
            elif b_ratio is not None:
                seg_score = b_ratio
            else:
                seg_score = None

            segments.append({
                "route":    route_name,
                "from":     a.get("name", a_code),
                "to":       b.get("name", b_code),
                "coords":   coords,
                "score":    seg_score,
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


# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Amtrak Underservice Map — Segment View</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0a0e17; color: #cbd5e1; font-family: 'DM Sans', system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; }

  #header {
    padding: 14px 20px 10px;
    background: linear-gradient(180deg, #0f1729 0%, #111827 100%);
    border-bottom: 1px solid #1e293b;
    display: flex; align-items: baseline; gap: 16px; flex-shrink: 0;
  }
  #header h1 { font-size: 15px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #f1f5f9; }
  #header p  { font-size: 12px; color: #475569; }

  #controls {
    padding: 8px 16px;
    background: #0f1729;
    border-bottom: 1px solid #1e293b;
    display: flex; gap: 8px; flex-shrink: 0; flex-wrap: wrap; align-items: center;
  }
  .filter-btn {
    padding: 5px 14px; border-radius: 20px;
    border: 1px solid #1e293b; background: transparent;
    color: #64748b; font-size: 12px; cursor: pointer;
    transition: all 0.15s; font-family: inherit; letter-spacing: 0.02em;
  }
  .filter-btn:hover { border-color: #475569; color: #cbd5e1; }
  .filter-btn.active { background: #1d4ed8; border-color: #1d4ed8; color: #fff; }

  .sep { width: 1px; height: 20px; background: #1e293b; margin: 0 4px; }

  #map { flex: 1; }

  #legend {
    position: absolute; bottom: 28px; right: 14px; z-index: 1000;
    background: rgba(15,23,41,0.95); border: 1px solid #1e293b;
    border-radius: 10px; padding: 14px 18px; font-size: 11px; line-height: 1.9;
    backdrop-filter: blur(8px);
  }
  #legend h3 { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #475569; margin-bottom: 6px; }
  .legend-section { margin-bottom: 10px; }
  .legend-row { display: flex; align-items: center; gap: 8px; color: #94a3b8; }
  .legend-dot  { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .legend-line { width: 22px; height: 3px; border-radius: 2px; flex-shrink: 0; }

  .leaflet-popup-content-wrapper {
    background: #1e293b !important; border: 1px solid #334155 !important;
    border-radius: 10px !important; box-shadow: 0 12px 32px rgba(0,0,0,0.6) !important;
    color: #e2e8f0 !important;
  }
  .leaflet-popup-tip { background: #1e293b !important; }
  .leaflet-popup-content { font-family: 'DM Sans', system-ui, sans-serif; margin: 12px 14px !important; }
  .popup-name  { font-size: 14px; font-weight: 700; margin-bottom: 8px; color: #f1f5f9; }
  .popup-row   { font-size: 12px; display: flex; justify-content: space-between; gap: 24px; margin: 3px 0; color: #94a3b8; }
  .popup-row span:last-child { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #e2e8f0; }
  .popup-label {
    font-size: 11px; font-weight: 600; margin-top: 8px; padding: 3px 10px;
    border-radius: 5px; display: inline-block;
  }

  .seg-popup-name  { font-size: 13px; font-weight: 700; margin-bottom: 4px; color: #f1f5f9; }
  .seg-popup-route { font-size: 11px; color: #64748b; margin-bottom: 8px; font-style: italic; }
  .seg-popup-row   { font-size: 11px; display: flex; justify-content: space-between; gap: 16px; margin: 2px 0; color: #94a3b8; }
  .seg-popup-row span:last-child { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #e2e8f0; }
</style>
</head>
<body>

<div id="header">
  <h1>Amtrak Underservice Map</h1>
  <p>Route segments colored by average endpoint demand ratio &mdash; thicker line = fewer weekly trips</p>
</div>

<div id="controls">
  <button class="filter-btn active" data-filter="all">All stations</button>
  <button class="filter-btn" data-filter="underserved">Underserved</button>
  <button class="filter-btn" data-filter="outperforms">Outperforms model</button>
  <button class="filter-btn" data-filter="top20">Top 20 targets</button>
  <div class="sep"></div>
  <button class="filter-btn active" data-seg="all_seg">All segments</button>
  <button class="filter-btn" data-seg="underserved_low">Underserved + low freq</button>
  <button class="filter-btn" data-seg="low_freq">Low freq (&le;14/wk)</button>
</div>

<div id="map"></div>

<div id="legend">
  <div class="legend-section">
    <h3>Station — actual ÷ predicted</h3>
    <div class="legend-row"><div class="legend-dot" style="background:#ef4444"></div><span>&lt; 0.33 &mdash; severely underserved</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#f97316"></div><span>0.33 &ndash; 0.5 &mdash; underserved</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#eab308"></div><span>0.5 &ndash; 0.77 &mdash; below model</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#22c55e"></div><span>0.77 &ndash; 1.3 &mdash; well-matched</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#06b6d4"></div><span>&gt; 1.3 &mdash; outperforms model</span></div>
    <div class="legend-row"><div class="legend-dot" style="background:#a855f7"></div><span>&lt; 1.0 + &le;14 trips/wk &mdash; underserved (strict)</span></div>
  </div>
  <div class="legend-section">
    <h3>Segment (avg of endpoints)</h3>
    <div class="legend-row"><div class="legend-line" style="background:#ef4444"></div><span>&lt; 0.33</span></div>
    <div class="legend-row"><div class="legend-line" style="background:#f97316"></div><span>0.33 &ndash; 0.5</span></div>
    <div class="legend-row"><div class="legend-line" style="background:#eab308"></div><span>0.5 &ndash; 0.77</span></div>
    <div class="legend-row"><div class="legend-line" style="background:#22c55e"></div><span>0.77 &ndash; 1.3</span></div>
    <div class="legend-row"><div class="legend-line" style="background:#06b6d4"></div><span>&gt; 1.3</span></div>
    <div class="legend-row" style="margin-top:4px;"><span style="color:#475569;">Thicker line = fewer weekly trips</span></div>
  </div>
</div>

<script>
const SEGMENTS = __SEGMENTS_JSON__;
const STATIONS = __STATIONS_JSON__;
const TOP20    = new Set(__TOP20_JSON__);

function scoreColor(s) {
  if (s == null) return '#334155';
  if (s < 0.33) return '#ef4444';   // strongly underserved
  if (s < 0.5)  return '#f97316';
  if (s < 0.77) return '#eab308';
  if (s <= 1.3) return '#22c55e';   // well-matched
  return '#06b6d4';                 // over model
}

function tripWeight(trips) {
  if (trips <= 14) return 5;
  if (trips <= 28) return 4;
  if (trips <= 56) return 3;
  if (trips <= 140) return 2.5;
  return 2;
}

// ── Map ───────────────────────────────────────────────────────────────────────
const map = L.map('map', { zoomControl: true, preferCanvas: true }).setView([38.5, -96.5], 5);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://carto.com/">CARTO</a> | <a href="https://www.openstreetmap.org/copyright">OSM</a>',
  subdomains: 'abcd', maxZoom: 19,
}).addTo(map);

// ── Segments ──────────────────────────────────────────────────────────────────
const segmentLayer = L.layerGroup().addTo(map);
let allSegments = [];

SEGMENTS.forEach(s => {
  const color   = scoreColor(s.score);
  const weight  = tripWeight(s.trips);
  const opacity = s.score != null ? 0.75 : 0.25;

  const line = L.polyline(s.coords, {
    color, weight, opacity, smoothFactor: 1.5, lineCap: 'round',
  });

  const fmtNum = n => n != null ? n.toLocaleString() : '\\u2014';
  const fmtR   = r => r != null ? r.toFixed(2) + '\\u00d7' : '\\u2014';

  line.bindPopup(`
    <div class="seg-popup-name">${s.from} \\u2192 ${s.to}</div>
    <div class="seg-popup-route">${s.route}</div>
    <div class="seg-popup-row"><span>Segment score</span><span>${fmtR(s.score)}</span></div>
    <div class="seg-popup-row"><span>Weekly trips</span><span>${s.trips}</span></div>
    <hr style="border:none;border-top:1px solid #334155;margin:6px 0;">
    <div class="seg-popup-row"><span>${s.from.split(',')[0]}</span><span>${fmtNum(s.from_rid)} riders \\u00b7 ${fmtR(s.from_r)}</span></div>
    <div class="seg-popup-row"><span>${s.to.split(',')[0]}</span><span>${fmtNum(s.to_rid)} riders \\u00b7 ${fmtR(s.to_r)}</span></div>
  `, { maxWidth: 300, minWidth: 240 });

  line._segData = s;
  allSegments.push(line);
  line.addTo(segmentLayer);
});

// ── Station markers ───────────────────────────────────────────────────────────
const stationLayer = L.layerGroup().addTo(map);
let allMarkers = [];

STATIONS.forEach(s => {
  const isSuppressed = s.ratio !== null && s.ratio < 1.0
                    && s.min_trips !== null && s.min_trips <= 14;
  const dotColor = isSuppressed ? '#a855f7' : s.color;

  const marker = L.circleMarker([s.lat, s.lon], {
    radius: s.radius, fillColor: dotColor,
    color: 'rgba(0,0,0,0.5)', weight: 1, opacity: 1, fillOpacity: 0.85,
  });

  const labelBg     = dotColor + '22';
  const labelBorder = dotColor + '66';
  marker.bindPopup(`
    <div class="popup-name">${s.name}</div>
    <div class="popup-row"><span>Annual ridership</span><span>${s.actual.toLocaleString()}</span></div>
    <div class="popup-row"><span>Model predicted</span><span>${s.predicted.toLocaleString()}</span></div>
    <div class="popup-label" style="color:${dotColor};background:${labelBg};border:1px solid ${labelBorder}">
      ${isSuppressed ? s.label + ' \u00b7 suppressed demand' : s.label}
    </div>
  `, { maxWidth: 240, minWidth: 200 });

  marker._stationData = s;
  allMarkers.push(marker);
  marker.addTo(stationLayer);
});

// ── Station filter controls ───────────────────────────────────────────────────
function applyStationFilter(filter) {
  stationLayer.clearLayers();
  allMarkers.forEach(m => {
    const s = m._stationData;
    let show = filter === 'all'
      || (filter === 'underserved' && s.ratio !== null && s.ratio < 1.0
          && s.min_trips !== null && s.min_trips <= 14)
      || (filter === 'outperforms' && s.ratio !== null && s.ratio > 1.3)
      || (filter === 'top20' && TOP20.has(s.name));
    if (show) m.addTo(stationLayer);
  });
}

document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn[data-filter]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyStationFilter(btn.dataset.filter);
  });
});

// ── Segment filter controls ───────────────────────────────────────────────────
function applySegmentFilter(filter) {
  segmentLayer.clearLayers();
  allSegments.forEach(line => {
    const s = line._segData;
    let show = filter === 'all_seg'
      || (filter === 'underserved_low' && s.score != null && s.score < 1.0 && s.trips <= 14)
      || (filter === 'low_freq'        && s.trips <= 14);
    if (show) line.addTo(segmentLayer);
  });
}

document.querySelectorAll('.filter-btn[data-seg]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn[data-seg]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applySegmentFilter(btn.dataset.seg);
  });
});
</script>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rail, rail_trips, stop_times, stops, shapes = load_gtfs()
    station_lookup, station_records, top20 = load_stations()
    segments = build_segments(rail, rail_trips, stop_times, stops, shapes, station_lookup)

    # Build station → min weekly trips from segment data
    station_min_trips: dict[str, int] = {}
    for seg in segments:
        for name in (seg["from"], seg["to"]):
            trips = seg["trips"]
            if name not in station_min_trips or trips < station_min_trips[name]:
                station_min_trips[name] = trips
    for rec in station_records:
        rec["min_trips"] = station_min_trips.get(rec["name"])

    # Serialise to JSON
    segments_json = json.dumps(segments, separators=(",", ":"))
    stations_json = json.dumps(station_records, separators=(",", ":"))
    top20_json    = json.dumps(top20, separators=(",", ":"))

    html = (
        HTML_TEMPLATE
        .replace("__SEGMENTS_JSON__", segments_json)
        .replace("__STATIONS_JSON__", stations_json)
        .replace("__TOP20_JSON__",    top20_json)
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")

    size_kb = OUT.stat().st_size / 1024
    print(f"\nWrote {OUT}  ({size_kb:.0f} KB)")
    assert size_kb < 2048, f"Output is {size_kb:.0f} KB — exceeds 2MB budget!"

    # Print top underserved segments — deduplicated by station pair (direction-agnostic)
    scored = [s for s in segments if s["score"] is not None]
    scored.sort(key=lambda s: s["score"])  # lowest ratio = most underserved first

    def dedup_by_pair(seg_list, n):
        seen = set()
        out = []
        for s in seg_list:
            key = frozenset([s["from"], s["to"]])
            if key not in seen:
                seen.add(key)
                out.append(s)
            if len(out) == n:
                break
        return out

    print("\nTop 10 underserved segments (by avg endpoint ratio):")
    for s in dedup_by_pair(scored, 10):
        print(f"  {s['route']:30s} {s['from']:>30s} → {s['to']:<30s} "
              f"score={s['score']:.2f}  trips/wk={s['trips']}")

    low_freq = [s for s in scored if s["trips"] <= 14]  # already sorted ascending
    print(f"\nTop 10 underserved + low-frequency (≤14/wk) segments:")
    for s in dedup_by_pair(low_freq, 10):
        print(f"  {s['route']:30s} {s['from']:>30s} → {s['to']:<30s} "
              f"score={s['score']:.2f}  trips/wk={s['trips']}")


if __name__ == "__main__":
    main()