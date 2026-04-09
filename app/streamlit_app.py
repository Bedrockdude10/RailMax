"""
streamlit_app.py

Interactive Streamlit app for the Amtrak Service Optimizer ridership model.

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

from data_loader import ALL_FEATURES, load_stations

MODEL_PATH = ROOT / "models" / "ebm_v1.pkl"


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
    st.set_page_config(page_title="Amtrak Service Optimizer", layout="wide")
    st.title("Amtrak Service Optimizer")
    st.caption("Trained on ~500 US Amtrak stations · Explainable Boosting Machine")

    ebm, feature_cols = load_model()
    df = load_data()

    if ebm is None:
        st.error(
            "No trained model found at `models/ebm_v1.pkl`. "
            "Run `python src/run_pipeline.py --only train` first."
        )
        st.stop()

    if feature_cols is None:
        feature_cols = ALL_FEATURES

    name_col = "station_name" if "station_name" in df.columns else "map_station"

    tab1, tab2 = st.tabs(["Existing Station Lookup", "Hypothetical Station"])

    # ── Tab 1: Existing stations ───────────────────────────────────────────────
    with tab1:
        st.subheader("Predicted vs. Actual Ridership")

        has_ridership = df["annual_ridership"].notna()
        available = df[has_ridership].copy()

        station_names = available[name_col].dropna().sort_values().tolist()
        selected = st.selectbox("Select a station", station_names)

        row = available[available[name_col] == selected].iloc[0]

        avail_features = [f for f in feature_cols if f in row.index]
        X = pd.DataFrame([row[avail_features].values], columns=avail_features)
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
            "Feature": avail_features,
            "Value": [row.get(f, np.nan) for f in avail_features],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── Tab 2: Hypothetical station ────────────────────────────────────────────
    with tab2:
        st.subheader("Hypothetical Station Prediction")
        st.caption("Adjust features below to estimate annual ridership at a proposed station.")

        with st.form("hypothetical_form"):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Location & Demographics**")
                metro_pop = st.number_input("Metro Population", value=500_000, min_value=0, step=50_000)
                dist_major = st.number_input(
                    "Distance to nearest major city (km)", value=50.0, min_value=0.0, step=10.0
                )
                num_nearby = st.number_input(
                    "Nearby Amtrak stations (within 80 km)", value=2, min_value=0, step=1
                )
                median_income = st.number_input(
                    "Median household income ($)", value=65_000, min_value=0, step=5_000
                )
                pct_transit = st.slider("% commuting by public transit", 0.0, 50.0, 5.0, step=0.5)
                pct_drove = st.slider("% driving alone to work", 0.0, 100.0, 75.0, step=1.0)

                st.markdown("**Service**")
                weekly_dep = st.number_input("Weekly departures", value=14, min_value=0, step=7)
                service_span = st.number_input("Service span (hours/day)", value=12.0, min_value=0.0, step=1.0)

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
                modes = st.slider("Total connecting modes", 1, 6, 2)

                st.markdown("**Other**")
                college_enrollment = st.number_input(
                    "College enrollment within 15 km", value=0, min_value=0, step=5_000
                )
                is_terminal = st.checkbox("Terminal station (end of line)")

            submitted = st.form_submit_button("Predict Ridership")

        if submitted:
            input_data = {
                "station_type": station_type,
                "has_heavy_rail": int(has_heavy_rail),
                "has_commuter_rail": int(has_commuter_rail),
                "has_light_rail": int(has_light_rail),
                "has_transit_bus": int(has_transit_bus),
                "has_intercity_bus": int(has_intercity_bus),
                "has_air_connection": int(has_air),
                "has_bikeshare": int(has_bike),
                "modes_served": modes,
                "metro_pop": metro_pop,
                "distance_to_nearest_major_city_km": dist_major,
                "num_nearby_stations": num_nearby,
                "weekly_departures": weekly_dep,
                "service_span_hours": service_span,
                "is_terminal": int(is_terminal),
                "median_household_income": median_income,
                "pct_public_transit": pct_transit,
                "pct_drove_alone": pct_drove,
                "college_enrollment_15km": college_enrollment,
            }

            row_df = pd.DataFrame([input_data])
            for col in feature_cols:
                if col not in row_df.columns:
                    row_df[col] = np.nan
            row_df = row_df[[f for f in feature_cols if f in row_df.columns]]

            pred_log = ebm.predict(row_df)[0]
            pred = int(np.expm1(pred_log))

            st.success(f"**Predicted annual ridership: {pred:,}**")
            st.caption(
                "Predicted on log scale then exponentiated back to ridership. "
                "See the README for known limitations, especially for large hub stations."
            )


if __name__ == "__main__":
    main()
