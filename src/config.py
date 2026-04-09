"""
config.py

Centralised constants for the RailMe pipeline.
Import from here instead of scattering magic numbers across modules.
"""

# ── Geographic matching thresholds ────────────────────────────────────────────

# parse_and_join.py: max distance (km) for coordinate-based map→NTAD match
PARSE_COORD_MATCH_KM = 30.0

# parse_and_join.py: minimum fuzzy name score (0–100) for map→NTAD match
PARSE_FUZZY_SCORE = 85

# build_gtfs_features.py: max distance (km) for GTFS stop → station coord match
GTFS_COORD_MATCH_KM = 2.0

# build_gtfs_features.py: minimum fuzzy name score for GTFS stop → station match
GTFS_FUZZY_SCORE = 80

# ── Feature computation parameters ────────────────────────────────────────────

# features.py: radius (km) within which a city counts for metro_pop (primary)
METRO_POP_RADIUS_KM = 50.0

# features.py: fallback radius (km) when no city found within primary radius
METRO_POP_FALLBACK_KM = 100.0

# features.py: minimum population to qualify as a "major city"
MAJOR_CITY_POP_THRESHOLD = 500_000

# features.py: radius (km) for counting nearby Amtrak stations
NEARBY_STATIONS_RADIUS_KM = 80.0

# NEC bounding box (approximate): Boston → Washington DC
# lat: 38.89°N (Union Station DC) to 42.36°N (South Station Boston)
# lon: east of -77.10°W
NEC_LAT_MIN = 38.89
NEC_LAT_MAX = 42.40
NEC_LON_MIN = -77.10  # stations must have lon >= this value

# build_gtfs_features.py: route length (km) threshold for pct_long_distance
LONG_DISTANCE_ROUTE_KM = 500.0

# build_college_features.py: radii (km) for college enrollment proximity
COLLEGE_RADII_KM = [15]

# ── Map display parameters (build_map.py) ─────────────────────────────────────

# Downsample GTFS shapes: keep every Nth point per segment
SHAPE_DOWNSAMPLE = 6

# Demand ratio thresholds for station/segment color coding
RATIO_SEVERE = 0.33
RATIO_UNDERSERVED = 0.5
RATIO_BELOW_MODEL = 0.77
RATIO_WELL_MATCHED_MAX = 1.3
RATIO_SUPPRESSED = 0.7       # threshold for "suppressed demand" label
SUPPRESSED_MAX_TRIPS = 14    # max weekly trips for "suppressed demand"

# Minimum actual ridership for top-20 underserved list
TOP20_MIN_RIDERSHIP = 50_000

# Station marker radius range (pixels)
MARKER_RADIUS_MIN = 5
MARKER_RADIUS_MAX = 22  # min + 17

# Stations co-located for display grouping: (display_name, [code1, code2, ...])
STATION_DISPLAY_GROUPS = [
    ("Boston South Station / Back Bay", ["BOS", "BBY"]),
    ("Newark / Newark Airport",         ["NWK", "EWR"]),
]

# ── State name → abbreviation ─────────────────────────────────────────────────

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