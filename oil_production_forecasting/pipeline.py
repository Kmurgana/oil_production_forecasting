#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: pipline.py
Description: Main pipeline for oil production forecasting.
Project: Oil Production Forecasting
Author: Kevin Murgana 
Created: 2025-05-17
Version: 1.0.0
"""

from oil_production_forecasting.dataset import load_feature_dataset, split_dataset
from oil_production_forecasting.features import generate_features
from oil_production_forecasting.modeling.train import train_lstm
from oil_production_forecasting.modeling.predict import LSTMPredictor

import pandas as pd
import json
import joblib
from pathlib import Path

from oil_production_forecasting.config import MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
import json

def main():
    print("🔁 Running full pipeline...")

    df = pd.read_csv(RAW_DATA_DIR / "oil_production_padd3_1960_present.csv", parse_dates=["period"])
    df = df.sort_values("period")
    df = generate_features(df)

    # Save feature data
    df.to_csv(PROCESSED_DATA_DIR / "oil_production_features.csv", index=False)

    # Split
    train_df, test_df = split_dataset(df)
    X_train = train_df.drop(columns=["period", "target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["period", "target"])
    y_test = test_df["target"]
    test_df[["period", "target"] + X_test.columns.tolist()].to_csv(PROCESSED_DATA_DIR / "test_features.csv", index=False)

    # Train
    config_path = MODELS_DIR / "lstm_best_params.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    model, x_scaler, y_scaler = train_lstm(X_train, y_train, config)

    model.save(MODELS_DIR / "lstm_model.h5")
    joblib.dump(x_scaler, MODELS_DIR / "scaler_X.pkl")
    joblib.dump(y_scaler, MODELS_DIR / "scaler_y.pkl")
    with open(MODELS_DIR / "lstm_config.json", "w") as f:
        json.dump(config, f)

    # Predict
    predictor = LSTMPredictor()
    y_pred = predictor.predict(X_test)

    df_out = pd.DataFrame({
        "ds": test_df["period"],
        "y_true": y_test,
        "y_pred": y_pred
    })
    df_out.to_csv(PROCESSED_DATA_DIR / "test_predictions.csv", index=False)

    print("✅ Full pipeline completed successfully.")

if __name__ == "__main__":
    main()
