#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: app.py
Description: Main Flask application for oil production forecasting.
Project: Oil Production Forecasting
Author: Kevin Murgana 
Created: 2025-05-17
Version: 1.0.0
"""

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from oil_production_forecasting.modeling.predict import LSTMPredictor
from oil_production_forecasting.config import PROCESSED_DATA_DIR

app = Flask(__name__, template_folder="templates")
predictor = LSTMPredictor()

# Must match your training feature order!
FEATURE_ORDER = [
    'MBBL', 'MBBL/D', 'year', 'month', 'quarter', 'dayofyear', 'weekofyear',
    'lag_1', 'lag_2', 'lag_3',
    'rolling_mean_3', 'rolling_mean_6',
    'rolling_std_3', 'rolling_std_6'
]
USER_FIELDS = ['MBBL/D', 'rolling_mean_3', 'month', 'quarter']
TIMESTEPS = [3, 2, 1]

def get_last_n_periods_defaults(n=3):
    df = pd.read_csv(PROCESSED_DATA_DIR / "oil_production_features.csv").sort_values("period")
    df.columns = [col.strip() for col in df.columns]
    missing = [feat for feat in FEATURE_ORDER if feat not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df_last = df.tail(n)
    defaults = {}
    for i, (_, row) in enumerate(df_last.iterrows(), 1):
        for feat in FEATURE_ORDER:
            defaults[f"{feat}_{n+1-i}"] = row[feat]
    return defaults

@app.route("/")
def home():
    defaults = get_last_n_periods_defaults(n=3)
    return render_template("index.html", defaults=defaults, user_fields=USER_FIELDS)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        defaults = get_last_n_periods_defaults(n=3)
        sequence = []
        for t in TIMESTEPS:
            feat_vec = []
            for feat in FEATURE_ORDER:
                key = f"{feat}_{t}"
                if feat in USER_FIELDS and key in request.form:
                    val = float(request.form[key])
                else:
                    val = defaults[key]
                feat_vec.append(val)
            sequence.append(feat_vec)
        X_input = np.array([sequence])  # shape (1, 3, n_features)
        y_pred = predictor.predict_sequence(X_input)
        prediction = round(float(y_pred[0]), 2)

        prod_hist = [sequence[i][FEATURE_ORDER.index("MBBL/D")] for i in range(3)]
        chart_labels = ["t-3", "t-2", "t-1", "Prediction"]
        chart_values = prod_hist + [prediction]

        for feat in USER_FIELDS:
            for t in TIMESTEPS:
                key = f"{feat}_{t}"
                if key in request.form:
                    defaults[key] = request.form[key]

        return render_template(
            "index.html",
            prediction=prediction,
            chart_labels=chart_labels,
            chart_values=chart_values,
            defaults=defaults,
            user_fields=USER_FIELDS
        )
    except Exception as e:
        defaults = get_last_n_periods_defaults(n=3)
        for feat in USER_FIELDS:
            for t in TIMESTEPS:
                key = f"{feat}_{t}"
                if key in request.form:
                    defaults[key] = request.form[key]
        return render_template("index.html", error=str(e), defaults=defaults, user_fields=USER_FIELDS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
