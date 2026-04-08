"""
streamlit_app.py

Interactive Streamlit app for the RailMe ridership prediction model.

Features:
1. Select an existing station → see predicted vs. actual ridership
2. Input features for a hypothetical station → get ridership estimate

Run with:
    streamlit run app/streamlit_app.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_loader import (
    ALL_FEATURES,
    BINARY_IPCD_FEATURES,
    NUMERIC_V0_FEATURES,
    load_stations,
)

MODEL_PATH = ROOT / "models" / "ebm_v0.pkl"


# ── Load resources (cached) ────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj["ebm"], obj["features"]
    return obj, None


@st.cache_data
def load_data():
    return load_stations()


# ── App ────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="RailMe — Ridership Predictor", layout="wide")
    st.title("RailMe: Rail Station Ridership Predictor")
    st.caption("v0 · Trained on US Amtrak stations · Explainable Boosting Machine")

    ebm, feature_cols = load_model()
    df = load_data()

    if ebm is None:
        st.error(
            "No trained model found at `models/ebm_v0.pkl`. "
            "Run `python src/train.py` first."
        )
        st.stop()

    if feature_cols is None:
        feature_cols = ALL_FEATURES

    name_col = "station_name" if "station_name" in df.columns else "Station"

    tab1, tab2 = st.tabs(["Existing Station Lookup", "Hypothetical Station"])

    # ── Tab 1: Existing stations ───────────────────────────────────────────────
    with tab1:
        st.subheader("Predicted vs. Actual Ridership")

        has_ridership = df["annual_ridership"].notna()
        available = df[has_ridership].copy()

        station_names = available[name_col].dropna().sort_values().tolist()
        selected = st.selectbox("Select a station", station_names)

        row = available[available[name_col] == selected].iloc[0]

        X = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
        pred_log = ebm.predict(X)[0]
        pred = int(np.expm1(pred_log))
        actual = int(row["annual_ridership"])
        pct_err = (pred - actual) / actual * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Actual Ridership", f"{actual:,}")
        col2.metric("Predicted Ridership", f"{pred:,}")
        col3.metric("Error", f"{pct_err:+.1f}%",
                    delta_color="inverse" if abs(pct_err) > 30 else "normal")

        st.divider()
        st.subheader("Station Features")
        feat_df = pd.DataFrame({
            "Feature": feature_cols,
            "Value": [row.get(f, np.nan) for f in feature_cols],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── Tab 2: Hypothetical station ────────────────────────────────────────────
    with tab2:
        st.subheader("Hypothetical Station Prediction")
        st.caption("Adjust features below to estimate annual ridership at a proposed station.")

        with st.form("hypothetical_form"):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Location**")
                lat = st.number_input("Latitude", value=40.0, min_value=24.0, max_value=50.0, step=0.1)
                lon = st.number_input("Longitude", value=-90.0, min_value=-125.0, max_value=-65.0, step=0.1)
                metro_pop = st.number_input("Metro Population", value=500_000, min_value=0, step=50_000)
                dist_major = st.number_input(
                    "Distance to nearest major city (km)", value=50.0, min_value=0.0, step=10.0
                )
                num_nearby = st.number_input(
                    "Nearby Amtrak stations (within 80 km)", value=2, min_value=0, step=1
                )
                is_nec = st.checkbox("Northeast Corridor station", value=False)
                is_metro = st.checkbox("In a metro area (CBSA)", value=True)
                modes = st.slider("Total connecting modes", 1, 6, 2)

            with c2:
                st.markdown("**Station Type**")
                station_type = st.selectbox(
                    "Station type",
                    [
                        "Station Building (with waiting room)",
                        "Platform with Shelter",
                        "Platform only (no shelter)",
                    ],
                )
                st.markdown("**Intermodal Connectivity**")
                has_heavy_rail = st.checkbox("Heavy rail / Metro connection")
                has_commuter_rail = st.checkbox("Commuter rail connection")
                has_light_rail = st.checkbox("Light rail / Tram connection")
                has_transit_bus = st.checkbox("Local transit bus")
                has_intercity_bus = st.checkbox("Intercity bus (Greyhound etc.)")
                has_air = st.checkbox("Airport served / nearby")
                has_bike = st.checkbox("Bikeshare available")

            submitted = st.form_submit_button("Predict Ridership")

        if submitted:
            input_data = {
                "lat": lat,
                "lon": lon,
                "station_type": station_type,
                "has_heavy_rail": int(has_heavy_rail),
                "has_commuter_rail": int(has_commuter_rail),
                "has_light_rail": int(has_light_rail),
                "has_transit_bus": int(has_transit_bus),
                "has_intercity_bus": int(has_intercity_bus),
                "has_air_connection": int(has_air),
                "has_bikeshare": int(has_bike),
                "modes_served": modes,
                "is_metro_area": int(is_metro),
                "metro_pop": metro_pop,
                "distance_to_nearest_major_city_km": dist_major,
                "num_nearby_stations": num_nearby,
                "is_northeast_corridor": int(is_nec),
            }

            row_df = pd.DataFrame([input_data])
            for col in feature_cols:
                if col not in row_df.columns:
                    row_df[col] = np.nan
            row_df = row_df[feature_cols]

            pred_log = ebm.predict(row_df)[0]
            pred = int(np.expm1(pred_log))

            st.success(f"**Predicted annual ridership: {pred:,}**")
            st.caption(
                "Predicted on log scale and exponentiated. "
                "This is a v0 model without service-frequency features — "
                "see PROJECT_PLAN.md for v1 roadmap."
            )


if __name__ == "__main__":
    main()
