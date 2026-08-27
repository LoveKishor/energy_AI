"""
Standalone training script — run this with the SAME Python environment
(appenv) that runs your Streamlit app, so the saved .pkl files are created
with the exact same scikit-learn version that will later load them.

Usage (from your appenv, in the energy_demand_prediction folder):
    appenv\\Scripts\\python.exe train_and_save.py        (Windows)
    ./appenv/bin/python train_and_save.py                (Mac/Linux)

Or simply, with appenv activated:
    python train_and_save.py

Requires household_energy_data.csv to be in the same folder.
Produces: consumption_model.pkl, peak_model.pkl
"""

import numpy as np
import pandas as pd
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(f"scikit-learn version: {sklearn.__version__}")
print(f"joblib version: {joblib.__version__}")

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df = pd.read_csv("household_energy_data.csv")
df["date"] = pd.to_datetime(df["date"])

FEATURE_COLS = ["people", "rooms", "outside_temp_c", "doy_sin", "doy_cos", "is_commercial"]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=frame.index)
    f["people"] = frame["people"]
    f["rooms"] = frame["rooms"]
    f["outside_temp_c"] = frame["outside_temp_c"]
    f["doy_sin"] = np.sin(2 * np.pi * frame["day_of_year"] / 365)
    f["doy_cos"] = np.cos(2 * np.pi * frame["day_of_year"] / 365)
    f["is_commercial"] = (frame["type"] == "Commercial").astype(int)
    return f[FEATURE_COLS]


X = build_features(df)
y_consumption = df["daily_consumption_kwh"]
y_peak = df["peak_power_kw"]

# ----------------------------------------------------------------------
# 2. Train/test split — by household, so test households are unseen
#
#    NOTE: household_ids is split manually (not via sklearn's
#    train_test_split) because pandas' newer Arrow-backed string dtype
#    is incompatible with sklearn's internal array indexing for this
#    kind of 1-D ID array on some pandas/sklearn version combinations
#    (raises: "TypeError: only integer scalar arrays can be converted
#    to a scalar index"). A manual shuffle+slice avoids that entirely.
# ----------------------------------------------------------------------
household_ids = np.array(df["household_id"].astype(str).unique(), dtype=object)

rng = np.random.default_rng(42)
rng.shuffle(household_ids)

n_test = max(1, int(len(household_ids) * 0.25))
test_ids = set(household_ids[:n_test])
train_ids = set(household_ids[n_test:])

train_mask = df["household_id"].astype(str).isin(train_ids)
test_mask = df["household_id"].astype(str).isin(test_ids)

X_train, X_test = X[train_mask], X[test_mask]
yc_train, yc_test = y_consumption[train_mask], y_consumption[test_mask]
yp_train, yp_test = y_peak[train_mask], y_peak[test_mask]


# ----------------------------------------------------------------------
# 3. Train & compare models
# ----------------------------------------------------------------------
def train_and_eval(X_train, X_test, y_train, y_test, target_name):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results[name] = {"model": model, "MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"[{target_name}] {name:20s}  MAE={mae:6.3f}  RMSE={rmse:6.3f}  R2={r2:.4f}")
    return results


print("=" * 70)
print("Daily Consumption (kWh)")
print("=" * 70)
consumption_results = train_and_eval(X_train, X_test, yc_train, yc_test, "Consumption")

print()
print("=" * 70)
print("Peak Power (kW)")
print("=" * 70)
peak_results = train_and_eval(X_train, X_test, yp_train, yp_test, "Peak Power")

best_consumption_name = max(consumption_results, key=lambda k: consumption_results[k]["R2"])
best_peak_name = max(peak_results, key=lambda k: peak_results[k]["R2"])
best_consumption_model = consumption_results[best_consumption_name]["model"]
best_peak_model = peak_results[best_peak_name]["model"]

print(f"\nBest model for consumption: {best_consumption_name}")
print(f"Best model for peak power:  {best_peak_name}")

# ----------------------------------------------------------------------
# 4. Save — created with THIS environment's sklearn version
# ----------------------------------------------------------------------
joblib.dump(best_consumption_model, "consumption_model.pkl")
joblib.dump(best_peak_model, "peak_model.pkl")

print(f"\nSaved consumption_model.pkl and peak_model.pkl")
print(f"(created with scikit-learn {sklearn.__version__} — load them only with this same version)")