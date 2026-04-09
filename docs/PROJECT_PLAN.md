# Amtrak Service Optimizer — Methodology

## Mission

Build an open-source, interpretable ML model that predicts passenger rail station ridership from publicly available data. The model identifies stations that are underserved relative to their potential demand, and estimates the ridership impact of increased service frequency.

The goal is to give transit advocates, planners, and journalists a credible, reproducible alternative to expensive consulting studies — one whose assumptions can be inspected, challenged, and improved.

---

## Model

**Method:** Explainable Boosting Machine (EBM) via [InterpretML](https://interpret.ml/)

An EBM is a modern gradient boosted generalized additive model (GAM). It learns a shape function for each feature — a nonlinear curve mapping feature value to its contribution to the prediction. Predictions are the sum of all shape function values plus a global intercept:

```
log(ridership) = intercept + f1(metro_pop) + f2(weekly_departures) + ... + f_n(x_n)
```

This means every prediction is fully decomposable: you can see exactly how much each feature is contributing for each station.

**Target variable:** `log1p(annual_ridership)` — annual station ridership (boardings + alightings), log-transformed to handle the heavy right skew of the distribution.

---

## Validation

**5-fold stratified group cross-validation.**

Stations are first clustered into 20 geographic groups via k-means on raw lat/lon. These groups are passed to `StratifiedGroupKFold` so that geographically proximate stations (e.g. Trenton and Princeton Junction) always appear in the same fold, preventing leakage. Stratification is on log-ridership quantiles so every fold sees the full range of station sizes.

After CV, a final model is fit on all data.

**Why no holdout set?** Out-of-fold predictions already provide an honest, per-station estimate for every station in the dataset — each station is predicted exactly once, by a model that never saw it during training. A separate holdout would waste signal from the largest and most informative stations.

---

## Features

### Intermodal connectivity (NTAD IPCD)
Binary flags for connecting services at each station: heavy rail/metro, commuter rail, light rail, transit bus, intercity bus, bikeshare, airport. Sourced from the National Transit Atlas Database Intermodal Passenger Connectivity Database.

### Geography
- `metro_pop`: population of the nearest US city within 50 km
- `distance_to_nearest_major_city_km`: Haversine distance to nearest city with population > 500k
- `num_nearby_stations`: count of other Amtrak stations within 80 km

### Service (GTFS)
Derived from the Amtrak GTFS feed:
- `weekly_departures`: total train departures per week
- `is_terminal`: whether the station is a route terminus
- `avg_dwell_time_sec`: average scheduled dwell time
- `service_span_hours`: hours between first and last departure
- `avg_stop_sequence_pct`: how far along its routes the station sits on average (0 = endpoint, 0.5 = midpoint)
- `avg_route_length_km`, `max_route_length_km`: route length statistics
- `pct_long_distance`: fraction of departures on long-distance routes (> 750 km)

### Demographics (ACS)
From the Census Bureau's American Community Survey 5-year estimates (2023), joined by county FIPS code:
- Commute mode shares: drove alone, public transit, rail, walked, work from home
- `median_household_income`

### Land use
- `college_enrollment_15km`: total college enrollment within 15 km (IPEDS 2023)
- `overseas_visitors_thousands`: international visitor volume for the station's metro area (NTTO/ITA 2024)

### Station
- `station_type`: Station Building (with waiting room) / Platform with Shelter / Platform only

---

## Underservice score

For each station, the model produces an out-of-fold prediction of log(ridership). The **demand ratio** is:

```
demand_ratio = actual_ridership / predicted_ridership
```

A ratio below 1.0 means the model expected more ridership than the station actually receives, suggesting the station may be underserved. A ratio above 1.0 means the station is outperforming its predicted potential.

Stations without GTFS data (non-Amtrak operated, see Limitations) are excluded from underservice rankings.

---

## What-if slider

The map includes a slider for adjusting weekly departures at each station. The estimated ridership impact is computed entirely in the browser from the EBM's pre-computed `weekly_departures` shape function:

```
delta = shape_score(new_trips) - shape_score(current_trips)
new_ridership = expm1(log1p(actual_ridership) + delta)
```

This anchors the estimate to observed ridership rather than the model's prediction, so the slider shows the marginal effect of service changes, not the model's absolute prediction.

**Caveat:** The shape function is learned cross-sectionally. Busy corridors tend to have both high frequency and high ridership, partly because demand drives supply. The slider estimates are illustrative — they show the historical cross-sectional association between frequency and ridership, not a controlled causal estimate.

---

## Known limitations

See README.md for the full list. Key issues:

1. **Major hub underprediction** — NYC, Chicago, Washington, and Boston are structural outliers that an additive model underfits. Their demand ratios are not reliable.

2. **GTFS-missing stations** — 44 non-Amtrak-operated stations (NJ Transit Atlantic City Line, Caltrans/SJRRC San Joaquin, VIA Rail Canada, CT Shore Line East, seasonal stations) have no service frequency data and weaker predictions.

3. **Cross-sectional causality** — the model captures correlations in the training data, not causal effects. Increasing service at a historically low-frequency station may not produce the ridership gains the shape function implies.

---

## Pipeline

```
parse_and_join   → data/processed/stations.csv
features         → adds metro_pop, distance, num_nearby_stations
build_gtfs       → adds GTFS schedule features
build_acs        → adds ACS commute + income features
build_college    → adds college enrollment proximity
add_tourism      → adds overseas visitor features
train            → EBM CV + final model + shape functions
build_map        → results/underservice_map.html
```

Run with: `python src/run_pipeline.py`
