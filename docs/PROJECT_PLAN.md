# HSR Ridership Forecasting Model — Project Plan

## Mission

Build an open-source, interpretable ML model that predicts passenger rail station ridership, trained on US Amtrak data and (later) French rail data. The model learns the relationship between station characteristics, intermodal connectivity, and ridership — which can then be used to estimate ridership at proposed HSR stations. This tool will help advocates, planners, and journalists generate credible, reproducible ridership estimates for proposed US high-speed rail corridors — challenging the expensive, opaque consulting models that currently dominate.

---

## Model Architecture

**Method:** Explainable Boosting Machine (EBM) via Microsoft InterpretML — a modern GAM that captures nonlinear feature effects and pairwise interactions while remaining fully interpretable. Not deep learning. Trains in seconds on tabular data.

**Unit of observation:** One station (e.g., Washington Union Station, Albany-Rensselaer, Whitefish MT)

**Target variable:** Annual station ridership (total boardings + alightings per year)

**Training data (v0):** ~500 US Amtrak stations from three joined datasets

**Validation:** Hold out major stations (Washington Union Station, Boston South Station, Philadelphia 30th Street, Chicago Union Station) and check whether the model predicts their ridership from features alone.

**Inference:** Predict ridership at proposed HSR station locations by plugging in their city features, connectivity profile, and hypothetical service characteristics.

---

## v0 Datasets (all uploaded)

| File | Contents | Rows | Key columns | Join key |
|------|----------|------|-------------|----------|
| `map_data.csv` | Amtrak station ridership 2025 | ~500 | Station, State, Latitude, Longitude, Value (annual ridership) | Station name + State |
| `NTAD_Amtrak_Stations.csv` | Station metadata | ~1000 | Code, StnType (TRAIN/BUS), City, State, lat, lon, StationName | StationName or Code |
| `NTAD_IPCD.csv` | Intermodal connectivity for all US passenger terminals | ~14000 | AMTRAKCODE, RAIL_H, RAIL_C, RAIL_LIGHT, RAIL_I, BUS_T, BUS_I, AIR_SERVE, BIKE_SHARE, FERRY_T, FERRY_I, MODES_SERV, CBSA_CODE, CBSA_TYPE | AMTRAKCODE |

**Encoding note:** `map_data.csv` is UTF-16LE with BOM (`\xff\xfe`). The other two are standard UTF-8 with BOM.

**Join strategy:**
1. Parse `map_data.csv` → ridership per station (name + state + lat/lon + annual ridership)
2. Parse `NTAD_Amtrak_Stations.csv` → filter to `StnType = TRAIN` → get station Code for each station
3. Join ridership to stations on station name + state (or nearest-coordinate fallback)
4. Join IPCD to stations on `AMTRAKCODE = Code`
5. Result: one row per Amtrak train station with ridership + metadata + connectivity features

---

## v0 Feature Set

All features computed programmatically from the three datasets plus a downloadable population dataset. No manual data entry.

### From the datasets directly

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `lat` | map_data.csv | Station latitude |
| 2 | `lon` | map_data.csv | Station longitude |
| 3 | `station_type` | NTAD Stations | Categorical: Station Building (with waiting room) / Platform with Shelter / Platform only |
| 4 | `has_heavy_rail` | IPCD `RAIL_H` | Binary: station connects to metro/subway |
| 5 | `has_commuter_rail` | IPCD `RAIL_C` | Binary: station connects to commuter rail |
| 6 | `has_light_rail` | IPCD `RAIL_LIGHT` | Binary: station connects to light rail/tram |
| 7 | `has_intercity_bus` | IPCD `BUS_I` | Binary: station connects to intercity bus (Greyhound etc.) |
| 8 | `has_transit_bus` | IPCD `BUS_T` | Binary: station connects to local transit bus |
| 9 | `has_air_connection` | IPCD `AIR_SERVE` | Binary: airport nearby or served |
| 10 | `has_bikeshare` | IPCD `BIKE_SHARE` | Binary: bikeshare available at station |
| 11 | `modes_served` | IPCD `MODES_SERV` | Count of total connecting modes |
| 12 | `is_metro_area` | IPCD `CBSA_TYPE` | Binary: station is in a CBSA metro area (vs. micro or rural) |

### Computed by Claude Code

| # | Feature | How it's computed |
|---|---------|-------------------|
| 13 | `metro_pop` | Join on city + state from a US cities population CSV (simplemaps or Census). Fallback: nearest city >50K within 50km. |
| 14 | `distance_to_nearest_major_city_km` | Haversine to nearest city with pop > 500K |
| 15 | `num_nearby_stations` | Count of other Amtrak TRAIN stations within 80km |
| 16 | `is_northeast_corridor` | Binary: lat/lon falls within NEC bounding box (roughly Boston 42.4°N to DC 38.9°N, east of -77.1°W) |

**Total: 16 features, ~400-500 training rows.** Good ratio for an EBM. The intermodal connectivity features (4-12) are the most interesting — they capture *why* some stations punch above their population weight (multimodal hubs attract more riders) and are directly relevant to HSR planning (proposed HSR stations at existing transit hubs will perform better).

---

## Versioning Strategy

### v0 — Programmatic only (Claude Code builds this)

Everything above. Train on US Amtrak stations with 16 features. Proof of concept.

### v1 — Add service characteristics and international data

| Feature | Source |
|---------|--------|
| `daily_departures` | Manual lookup from Amtrak timetables |
| `num_routes_served` | Manual count per station |
| `avg_speed_fastest_route_kmh` | Manual lookup |
| `fare_per_km_to_nearest_hub` | Manual lookup from Amtrak.com |
| `car_ownership_rate` | FHWA, joined by state |
| France ART station data | Download from autorite-transports.fr |

### v2 — Corridor-level ridership prediction

- Define a proposed HSR route as a sequence of station locations
- Predict ridership at each proposed station using v1 model with hypothetical HSR service features
- Sum station predictions to get corridor-level ridership estimate
- Interactive Streamlit tool where user draws a corridor and adjusts service parameters
- This is the original project goal — v0 and v1 are the foundation

---

## Claude Code Handoff Instructions

### Prompt for Claude Code

> Set up a Python project for an ML-based rail station ridership prediction model. Read the project plan at `docs/PROJECT_PLAN.md` for full context.
>
> The raw data files are in `data/raw/`:
> - `map_data.csv` — Amtrak station ridership 2025 (UTF-16LE encoded with BOM). Columns: Latitude, Longitude, Note, State, Station, Value
> - `NTAD_Amtrak_Stations.csv` — Amtrak station metadata (UTF-8 with BOM). Key columns: Code, StnType, City, State, lat, lon, StationName
> - `NTAD_IPCD.csv` — Intermodal Passenger Connectivity Database (UTF-8 with BOM). Key columns: AMTRAKCODE, RAIL_H, RAIL_C, RAIL_LIGHT, BUS_T, BUS_I, AIR_SERVE, BIKE_SHARE, MODES_SERV, CBSA_TYPE. Note: IPCD contains ~14K rows for all terminal types — filter to rows where AMTRAKCODE is not null to get Amtrak stations.
>
> For v0, build:
> 1. Data parsing and joining:
>    - Parse map_data.csv (handle UTF-16LE), extract station name, state, lat, lon, annual ridership
>    - Parse NTAD stations, filter to StnType=TRAIN
>    - Join ridership to stations on station name + state (fuzzy match or nearest-coordinate fallback)
>    - Filter IPCD to rows with non-null AMTRAKCODE, join to stations on AMTRAKCODE = Code
>    - Output: `data/processed/stations.csv` with one row per station
>
> 2. Feature computation:
>    - Extract IPCD binary features: RAIL_H, RAIL_C, RAIL_LIGHT, BUS_T, BUS_I, AIR_SERVE, BIKE_SHARE → convert coded values to binary (1 = service present, see IPCD coding scheme)
>    - Extract MODES_SERV and CBSA_TYPE from IPCD
>    - Compute metro_pop by joining from a US cities population dataset (download simplemaps free US cities CSV or bundle one)
>    - Compute distance_to_nearest_major_city_km via Haversine
>    - Compute num_nearby_stations (count of other stations within 80km)
>    - Compute is_northeast_corridor binary flag
>
> 3. Standardized data loader (`src/data_loader.py`):
>    - Reads stations.csv, handles NaN for features added in v1
>    - Returns clean DataFrame ready for EBM
>
> 4. EBM training pipeline (`src/train.py`):
>    - Log-transform target variable (station ridership is heavily right-skewed)
>    - Hold out specific stations for validation: Washington Union Station, Boston South Station, Philadelphia 30th Street, Chicago Union Station
>    - Fit ExplainableBoostingRegressor from interpret.glassbox
>    - Output shape function plots to results/shape_functions/
>    - Save trained model to models/ebm_v0.pkl
>    - Print validation metrics: RMSE, R², held-out predictions vs actuals
>
> 5. Validation script (`src/validate.py`):
>    - Load held-out stations, predict, compare to actuals
>    - Output error report to results/metrics/
>
> 6. Skeleton Streamlit app (`app/streamlit_app.py`):
>    - Loads pre-trained model
>    - User can select an existing station to see predicted vs actual ridership
>    - User can input features for a hypothetical station to get prediction
>
> 7. Repo hygiene:
>    - Clear Jupyter notebook outputs before committing
>    - Virtual environment with requirements.txt pinning all versions
>    - .gitignore for data/raw/ if large, .ipynb_checkpoints, __pycache__, models/*.pkl

### Repo Structure

```
hsr-ridership-model/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── map_data.csv                    # Amtrak ridership 2025
│   │   ├── NTAD_Amtrak_Stations.csv        # Station metadata
│   │   └── NTAD_IPCD.csv                   # Intermodal connectivity
│   └── processed/
│       └── stations.csv                    # Joined + feature-enriched training data
├── src/
│   ├── data_loader.py                      # Standardized data loading, NaN handling
│   ├── parse_and_join.py                   # Parse all 3 CSVs, join, filter, output stations.csv
│   ├── features.py                         # Population join, Haversine, nearby count, NEC flag
│   ├── train.py                            # EBM training pipeline
│   └── validate.py                         # Hold-out validation
├── app/
│   └── streamlit_app.py                    # Interactive prediction tool
├── models/
│   └── ebm_v0.pkl                          # Saved trained model
├── results/
│   ├── shape_functions/                    # PNGs of feature shape plots
│   └── metrics/                            # Validation metrics
├── notebooks/
│   └── eda.ipynb                           # Exploratory data analysis
└── docs/
    ├── PROJECT_PLAN.md                     # This file
    └── experiments.md                      # Log of experiments and results
```

---

## IPCD Coding Scheme

The IPCD uses numeric codes, not simple booleans. Claude Code needs to decode these:

| Value | Meaning |
|-------|---------|
| 1 | Service available at this facility |
| 2 | Service available nearby (within buffer) |
| 3 | Service not available |
| 0 | Not applicable / not evaluated |

**For our features:** Code `1` or `2` → binary `1` (service accessible). Code `3` or `0` → binary `0`.

`MODES_SERV` is already a count (1-6 typically). `CBSA_TYPE`: `1` = metro, `2` = micro, `0` = rural.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Station ridership is dominated by NEC mega-stations creating extreme skew | Model fits big stations, poorly predicts small ones | Log-transform target; EBM handles nonlinearity naturally |
| IPCD AMTRAKCODE join misses some stations | Lost rows | Fall back to nearest-coordinate match between IPCD and station locations |
| Population join fails for small towns | Missing metro_pop feature | Nearest-city-within-radius fallback; NaN handling in data loader |
| v0 has no service characteristics (frequency, speed, fare) | Can't distinguish "busy because good service" from "busy because big city" | Expected for v0; service features added in v1 |
| US-only training doesn't generalize to international HSR | Can't predict French/Japanese ridership | France data added in v1; car_ownership and country features capture structural differences |

---

## Success Criteria

**v0:**
1. Model trains and produces shape function plots without errors
2. Shape functions are directionally correct: ridership increases with population, modes_served, heavy rail connectivity; higher in NEC; higher for station buildings vs platforms
3. Held-out major station predictions within 30% of actual ridership
4. Pipeline is end-to-end reproducible from raw data to trained model

**v1:**
1. Held-out predictions within 20% with service features added
2. France + US model shows coherent cross-country patterns
3. Streamlit app is usable by a non-technical person

**v2:**
1. Corridor-level ridership estimates for proposed US HSR routes
2. Interactive scenario tool with adjustable service parameters
3. Results credible enough to share with HSR advocacy community

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Data processing | Python, Pandas |
| Distance computation | geopy (Haversine) |
| Model | InterpretML (ExplainableBoostingRegressor) |
| Visualization | Plotly for shape functions |
| Interactive tool | Streamlit |
| Version control | Git / GitHub |
| Data storage | CSV in repo |