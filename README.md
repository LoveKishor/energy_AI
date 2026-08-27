# Household Energy Predictor

A Streamlit web application for predicting daily electricity consumption and peak power demand based on household characteristics and environmental conditions.

## 📋 Overview

This application predicts two key energy metrics:
- **Daily electricity consumption** (kWh)
- **Peak power demand** (kW)

The prediction engine considers:
- Building type (Residential/Commercial)
- Number of occupants
- Number of rooms
- Date/season
- Outside temperature

## 🚀 Features

### Flexible Input Options
- **Fixed household profile**: Define building type, people count, and rooms
- **Date input**: Single day or date range
- **Temperature input**: Single value, comma-separated series, or CSV upload

### Output Display
- Single-day predictions with metric cards
- Time-series predictions with line charts
- Peak consumption and peak power identification
- Downloadable results in CSV format


### Using the App

1. **Household Profile** (left panel):
   - Select building type (Residential/Commercial)
   - Enter number of people (1-100)
   - Enter number of rooms (1-50)

2. **Date & Temperature** (right panel):
   - Choose date input mode (Single date or Date range)
   - Select temperature input method:
     - Single value (applies to all dates)
     - Comma-separated series
     - Upload CSV file with 'temp' column

3. Click **🔮 Predict** to see results

## 📊 Model Integration

### Using Your Own Trained Models

The app supports both a built-in hypothetical model and real trained models:

1. Train your models (e.g., with scikit-learn)
2. Save them using joblib:
3. Place the `.pkl` files in the same directory as `energy_app.py`
4. The app will automatically detect and use your models

### Feature Engineering

The app expects features in this exact order:
- `people`: Number of occupants
- `rooms`: Number of rooms
- `outside_temp_c`: Temperature in Celsius
- `doy_sin`: Day of year sine transformation
- `doy_cos`: Day of year cosine transformation
- `is_commercial`: Binary indicator (1 for Commercial, 0 for Residential)

## 📁 File Structure

```
energy-ai/
├── energy_app.py           # Main application
├── requirements.txt        # Python dependencies
├── consumption_model.pkl   # (Optional) Trained consumption model
├── peak_model.pkl          # (Optional) Trained peak model
└── README.md              # This file
```
