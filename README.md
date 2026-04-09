# Amtrak Service Optimizer

An open-source, interpretable machine learning model that predicts annual ridership at US Amtrak stations and identifies where increased service would have the greatest impact.

**[View the interactive map](https://bedrockdude10.github.io/RailMax/results/underservice_map.html)**

---

## What this is

A data-driven tool for understanding which Amtrak stations are underserved relative to their potential demand — and by how much. The model learns the relationship between station characteristics (city population, intermodal connections, service frequency, neighborhood demographics) and annual ridership, then flags stations where actual ridership falls short of what the model expects.

The map lets you:
- Explore every Amtrak station colored by its demand ratio (actual vs. predicted ridership)
- Adjust weekly departures at any station with a slider to estimate the ridership impact
- See the top 20 city pairs most likely to benefit from increased service frequency

## Model

**Method:** [Explainable Boosting Machine (EBM)](https://interpret.ml/) — a modern gradient boosted GAM that captures nonlinear feature effects while remaining fully interpretable. Every prediction is a sum of per-feature shape functions you can inspect directly.

**Target:** Annual station ridership (boardings + alightings)

**Training data:** ~500 US Amtrak stations

**Validation:** 5-fold stratified group cross-validation. Stations are clustered into 20 geographic groups via k-means on lat/lon, and groups are kept together across folds to prevent leakage from nearby stations (e.g. Trenton and Princeton Junction are always in the same fold).

**CV metrics (out-of-fold):**

| Metric | Value |
|--------|-------|
| R² | 0.76 |
| RMSE (log scale) | 0.84 |

## Features

| Category | Features |
|----------|----------|
| Geography | Metro population, distance to nearest major city, number of nearby Amtrak stations |
| Intermodal connectivity | Heavy rail, commuter rail, light rail, transit bus, intercity bus, bikeshare, air connection, total modes served |
| Service | Weekly departures, service span (hours), terminal status, avg dwell time, route length, % long-distance routes |
| Demographics | % drive alone, % public transit commuters, % rail commuters, % work from home, median household income |
| Land use | College enrollment within 15 km, overseas visitors (top-50 tourist metros) |
| Station | Station type (building / platform with shelter / platform only) |

## Known limitations

**Major hub underprediction.** The model significantly underestimates ridership at the largest stations:

| Station | Actual | Predicted | Error |
|---------|--------|-----------|-------|
| New York Penn | 13,037,414 | 4,169,338 | −68% |
| Chicago Union | 3,175,856 | 513,457 | −84% |
| Washington Union | 6,010,221 | 3,164,448 | −47% |
| Philadelphia 30th St | 5,586,174 | 4,888,767 | −13% |

These stations sit far outside the distribution the model was trained on. An additive model cannot fully capture what makes Penn Station NYC unique — it's not just in a big city, it's the convergence point for the NEC, Empire Service, Lake Shore Limited, and long-distance routes serving the entire Eastern seaboard. The underservice rankings for these stations should be interpreted with caution.

**GTFS-missing stations.** 44 stations have no GTFS data and therefore no service frequency features. These are not Amtrak-operated stations — they are run by NJ Transit (Atlantic City Line), Caltrans/SJRRC (San Joaquin/ACE corridor), VIA Rail Canada, CT Shore Line East, and a handful of seasonal/event stations. Their predictions rely on demographic and connectivity features only and are less reliable.

**Cross-sectional causality.** The `weekly_departures` shape function is learned from cross-sectional data, meaning the causal arrow is partly reversed: high-ridership corridors tend to get more service because they're already busy. The what-if slider estimates are illustrative, not causal forecasts.

## Repo structure

```
├── src/
│   ├── run_pipeline.py          # End-to-end orchestrator
│   ├── parse_and_join.py        # Join raw CSVs → stations.csv
│   ├── features.py              # Metro pop, distance, nearby stations
│   ├── build_gtfs_features.py   # GTFS schedule features
│   ├── build_acs_features.py    # ACS commute + income features
│   ├── build_college_features.py
│   ├── add_tourism_features.py  # Overseas visitor features
│   ├── data_loader.py           # Standardised loader + feature definitions
│   ├── train.py                 # EBM cross-validation + final model
│   └── build_map.py             # Generate underservice_map.html
├── app/
│   └── streamlit_app.py         # Interactive prediction tool
├── data/
│   ├── raw/                     # Source data (not in git)
│   └── processed/
│       └── stations.csv         # Feature-enriched training data
├── results/
│   ├── underservice_map.html    # Interactive map (GitHub Pages)
│   └── metrics/                 # OOF predictions, training metrics, shape data
├── models/                      # Trained model (not in git — retrain locally)
└── docs/
    └── PROJECT_PLAN.md
```

## Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (parse → features → train → map)
python src/run_pipeline.py

# Or resume from a specific step
python src/run_pipeline.py --from train

# Launch the Streamlit app
streamlit run app/streamlit_app.py
```

The pipeline requires the raw data files in `data/raw/`. The trained model (`models/ebm_v1.pkl`) is not committed to the repo — run the pipeline to regenerate it.

## Data sources

| Dataset | Source |
|---------|--------|
| Station ridership | Amtrak (2024–2025) |
| Station metadata | NTAD Amtrak Stations |
| Intermodal connectivity | NTAD Intermodal Passenger Connectivity Database (IPCD) |
| GTFS schedules | Amtrak GTFS feed |
| Demographics | US Census ACS 5-year estimates (2023), tables B08301, B19013 |
| City populations | US Census / simplemaps |
| College enrollment | IPEDS (2023) |
| International visitors | NTTO/ITA Top-50 Tourist Markets (2024) |

## License

MIT
