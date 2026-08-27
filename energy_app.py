"""
Household Energy Prediction App
================================
Streamlit app that predicts:
    - Daily electricity consumption (kWh)
    - Peak power demand (kW)

Independent variables:
    - type (Residential / Commercial)   -> always a SINGLE value
    - people                             -> always a SINGLE value
    - rooms                              -> always a SINGLE value
    - date                                -> SINGLE value OR a SERIES (date range)
    - outside_temp_c                      -> SINGLE value OR a SERIES

Rule implemented: if date OR temperature is given as a series, every other
variable is broadcast (repeated) to match that series length, so the model
always receives a clean n-row table.

--------------------------------------------------------------------------
HOW TO SWAP IN YOUR OWN TRAINED MODEL
--------------------------------------------------------------------------
This file currently uses a HYPOTHETICAL rule-based model (see the
`HypotheticalModel` class below) so the app runs standalone with no
dependencies beyond streamlit/pandas/numpy.

To use your real trained models instead:
    1. Save your trained sklearn models with joblib, e.g.:
           import joblib
           joblib.dump(consumption_model, "consumption_model.pkl")
           joblib.dump(peak_model, "peak_model.pkl")
    2. Place the .pkl files next to this app.py file.
    3. In `load_models()` below, uncomment the joblib-loading block and
       delete/comment out the HypotheticalModel block.
    4. Make sure `build_features()` matches EXACTLY the feature engineering
       you used at training time (same column order, same encoding).
--------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Household Energy Predictor", layout="wide")

# ==========================================================================
# 1. MODEL LAYER  (replace this section with your real trained model)
# ==========================================================================

class HypotheticalModel:
    """
    Stand-in for a trained ML model. Mimics the same kind of logic used to
    fabricate the training data (U-shaped temperature response, type/people/
    rooms effects, seasonal cycle) so the app is demo-able end-to-end.
    Replace with a real joblib-loaded sklearn/XGBoost model for production.
    """

    def __init__(self, target: str):
        assert target in ("consumption", "peak")
        self.target = target

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        is_comm = X["is_commercial"].values
        people = X["people"].values
        rooms = X["rooms"].values
        temp = X["outside_temp_c"].values
        doy_sin = X["doy_sin"].values
        doy_cos = X["doy_cos"].values

        base_load = np.where(is_comm == 1, 11.0, 4.0)
        people_factor = np.where(is_comm == 1, 0.28, 0.75)
        room_factor = np.where(is_comm == 1, 0.55, 0.35)

        heating = np.maximum(0, 18 - temp) ** 1.3
        cooling = np.maximum(0, temp - 30) ** 1.15
        temp_eff = np.where(
            is_comm == 1,
            0.09 * heating + 0.22 * cooling,
            0.16 * heating + 0.10 * cooling,
        )

        # mild seasonal wiggle from cyclical day-of-year encoding
        season_eff = 0.5 * doy_sin + 0.3 * doy_cos

        daily_consumption = (
            base_load + people_factor * people + room_factor * rooms + temp_eff + season_eff
        )
        daily_consumption = np.maximum(0.5, daily_consumption)

        if self.target == "consumption":
            return np.round(daily_consumption, 2)

        # peak power derived from consumption via a load factor
        load_factor = np.where(
            is_comm == 1,
            np.clip(0.50 + 0.003 * np.minimum(people, 50), 0.48, 0.68),
            np.clip(0.34 + 0.01 * np.minimum(people, 8), 0.28, 0.48),
        )
        peak_power = np.maximum(0.3, (daily_consumption / 24) / load_factor)
        return np.round(peak_power, 2)


import os
import joblib

CONSUMPTION_MODEL_PATH = "consumption_model.pkl"
PEAK_MODEL_PATH = "peak_model.pkl"


@st.cache_resource
def load_models():
    """
    Auto-detects trained models saved from the notebook (via joblib.dump)
    in the same folder as this app. If both .pkl files are found, loads
    and uses them. Otherwise falls back to the HypotheticalModel so the
    app still runs standalone.

    To use your real models: run the notebook's Section 6, which saves
        consumption_model.pkl
        peak_model.pkl
    then copy those two files into the same folder as this app.py.
    """
    if os.path.exists(CONSUMPTION_MODEL_PATH) and os.path.exists(PEAK_MODEL_PATH):
        consumption_model = joblib.load(CONSUMPTION_MODEL_PATH)
        peak_model = joblib.load(PEAK_MODEL_PATH)
        using_real_models = True
    else:
        consumption_model = HypotheticalModel("consumption")
        peak_model = HypotheticalModel("peak")
        using_real_models = False

    return consumption_model, peak_model, using_real_models


consumption_model, peak_model, using_real_models = load_models()

FEATURE_COLS = ["people", "rooms", "outside_temp_c", "doy_sin", "doy_cos", "is_commercial"]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw input columns into the exact feature matrix the models expect.
    MUST match the feature engineering used at training time.
    """
    f = pd.DataFrame(index=raw.index)
    f["people"] = raw["people"]
    f["rooms"] = raw["rooms"]
    f["outside_temp_c"] = raw["outside_temp_c"]
    f["doy_sin"] = np.sin(2 * np.pi * raw["day_of_year"] / 365)
    f["doy_cos"] = np.cos(2 * np.pi * raw["day_of_year"] / 365)
    f["is_commercial"] = (raw["type"] == "Commercial").astype(int)
    return f[FEATURE_COLS]


# ==========================================================================
# 2. CORE PREDICTION FUNCTION
#    Handles the "single value OR series, broadcast to match" rule.
# ==========================================================================

def predict_series(house_type: str, people: int, rooms: int, dates, temps) -> pd.DataFrame:
    """
    house_type, people, rooms : SINGLE fixed values for the household.
    dates : a single date or a list/array of dates.
    temps : a single temperature or a list/array of temperatures.

    If either dates or temps is a series, the other inputs (including a
    single-value date or temp) are broadcast/repeated to the same length.
    """
    dates = pd.to_datetime(pd.Series(np.atleast_1d(dates)))
    temps = pd.Series(np.atleast_1d(temps)).astype(float)

    n = max(len(dates), len(temps))

    if len(dates) == 1 and n > 1:
        dates = pd.Series([dates.iloc[0]] * n)
    if len(temps) == 1 and n > 1:
        temps = pd.Series([temps.iloc[0]] * n)

    if len(dates) != len(temps):
        raise ValueError(
            f"dates (len={len(dates)}) and temps (len={len(temps)}) must match "
            f"in length, or one of them must be a single value."
        )

    raw = pd.DataFrame({
        "type": [house_type] * n,
        "people": [people] * n,
        "rooms": [rooms] * n,
        "day_of_year": dates.dt.dayofyear.values,
        "outside_temp_c": temps.values,
    })

    X = build_features(raw)

    pred_consumption = consumption_model.predict(X)
    pred_peak = peak_model.predict(X)

    out = pd.DataFrame({
        "date": dates.dt.date.values,
        "outside_temp_c": temps.values,
        "predicted_consumption_kwh": pred_consumption,
        "predicted_peak_kw": pred_peak,
    })
    return out


def summarize_peaks(series_df: pd.DataFrame) -> dict:
    peak_c = series_df.loc[series_df["predicted_consumption_kwh"].idxmax()]
    peak_p = series_df.loc[series_df["predicted_peak_kw"].idxmax()]
    return {
        "peak_consumption_date": str(peak_c["date"]),
        "peak_consumption_value_kwh": float(peak_c["predicted_consumption_kwh"]),
        "peak_power_date": str(peak_p["date"]),
        "peak_power_value_kw": float(peak_p["predicted_peak_kw"]),
    }


# ==========================================================================
# 3. STREAMLIT UI
# ==========================================================================

st.title("🏠 Household Energy Predictor")

if using_real_models:
    st.success(
        f"✅ Using your trained models (`{CONSUMPTION_MODEL_PATH}`, `{PEAK_MODEL_PATH}`).",
        icon="✅",
    )
else:
    st.warning(
        "⚠️ No trained model files found — running on a **placeholder** rule-based "
        f"model. Run the notebook's Section 6 to generate `{CONSUMPTION_MODEL_PATH}` "
        f"and `{PEAK_MODEL_PATH}`, then place them next to `app.py` and refresh.",
        icon="⚠️",
    )

st.divider()

col_fixed, col_variable = st.columns([1, 1.4])

# --------------------------------------------------------------------
# FIXED (household) inputs — always single values
# --------------------------------------------------------------------
with col_fixed:
    st.subheader("Household profile (fixed)")
    house_type = st.selectbox("Type", ["Residential", "Commercial"])
    people = st.number_input("Number of people", min_value=1, max_value=100, value=4, step=1)
    rooms = st.number_input("Number of rooms", min_value=1, max_value=50, value=5, step=1)

# --------------------------------------------------------------------
# VARIABLE inputs — date & temperature, each single value OR series
# --------------------------------------------------------------------
with col_variable:
    st.subheader("Date & Temperature (single value or series)")

    date_mode = st.radio("Date input mode", ["Single date", "Date range (series)"], horizontal=True)

    if date_mode == "Single date":
        single_date = st.date_input("Date", value=pd.Timestamp.today())
        dates_input = [single_date]
    else:
        d1, d2 = st.columns(2)
        start_date = d1.date_input("Start date", value=pd.Timestamp("2026-01-01"))
        end_date = d2.date_input("End date", value=pd.Timestamp("2026-12-31"))
        if start_date > end_date:
            st.error("Start date must be before end date.")
            st.stop()
        dates_input = pd.date_range(start_date, end_date, freq="D")
        st.caption(f"→ {len(dates_input)} days in this range")

    temp_mode = st.radio(
        "Temperature input mode",
        ["Single value (applies to all dates)", "Series (comma-separated)", "Series (upload CSV)"],
        horizontal=False,
    )

    if temp_mode == "Single value (applies to all dates)":
        single_temp = st.number_input("Outside temperature (°C)", value=20.0, step=0.5)
        temps_input = [single_temp]

    elif temp_mode == "Series (comma-separated)":
        st.caption(
            f"Enter exactly {len(dates_input)} comma-separated values "
            f"(one per date), e.g. 12.5, 13.0, 14.2, ..."
        )
        raw_text = st.text_area("Temperatures (°C)", height=100)
        if raw_text.strip():
            try:
                temps_input = [float(x.strip()) for x in raw_text.split(",") if x.strip() != ""]
            except ValueError:
                st.error("Could not parse temperatures — make sure they're comma-separated numbers.")
                st.stop()
        else:
            temps_input = []

    else:  # Upload CSV
        st.caption("CSV must have a single column named 'temp' with one row per date.")
        uploaded = st.file_uploader("Upload temperature CSV", type=["csv"])
        if uploaded is not None:
            temp_df = pd.read_csv(uploaded)
            if "temp" not in temp_df.columns:
                st.error("CSV must contain a column named 'temp'.")
                st.stop()
            temps_input = temp_df["temp"].tolist()
        else:
            temps_input = []

st.divider()

# --------------------------------------------------------------------
# RUN PREDICTION
# --------------------------------------------------------------------
run = st.button("🔮 Predict", type="primary", use_container_width=True)

if run:
    if len(temps_input) == 0:
        st.error("Please provide at least one temperature value.")
        st.stop()

    n_dates = len(dates_input)
    n_temps = len(temps_input)

    # Validate: one of them must be length 1, or both must match
    if n_dates != 1 and n_temps != 1 and n_dates != n_temps:
        st.error(
            f"Date series has {n_dates} entries but temperature series has "
            f"{n_temps} entries. One of them must be a single value, or both "
            f"must be the same length."
        )
        st.stop()

    try:
        result_df = predict_series(
            house_type=house_type,
            people=people,
            rooms=rooms,
            dates=dates_input,
            temps=temps_input,
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Results")

    if len(result_df) == 1:
        r = result_df.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Date", str(r["date"]))
        m2.metric("Predicted daily consumption", f"{r['predicted_consumption_kwh']:.2f} kWh")
        m3.metric("Predicted peak power", f"{r['predicted_peak_kw']:.2f} kW")
    else:
        peaks = summarize_peaks(result_df)
        m1, m2 = st.columns(2)
        m1.metric(
            "📈 Peak consumption day",
            peaks["peak_consumption_date"],
            f"{peaks['peak_consumption_value_kwh']:.2f} kWh",
        )
        m2.metric(
            "⚡ Peak power day",
            peaks["peak_power_date"],
            f"{peaks['peak_power_value_kw']:.2f} kW",
        )

        chart_df = result_df.set_index("date")[["predicted_consumption_kwh", "predicted_peak_kw"]]
        st.line_chart(chart_df)

    with st.expander("Full prediction table"):
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            "Download predictions as CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="household_energy_predictions.csv",
            mime="text/csv",
        )
else:
    st.info("Set your inputs above and click **Predict**.")