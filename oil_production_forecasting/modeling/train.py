#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: train.py
Description: Train LSTM model for oil production forecasting.
Project: Oil Production Forecasting
Author: Kevin Murgana 
Created: 2025-05-17
Version: 1.0.0
"""
from dotenv import load_dotenv
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import json
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from oil_production_forecasting.config import MODELS_DIR, PROCESSED_DATA_DIR

# For pretty metrics output
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

app = typer.Typer()

def create_sequences(features: np.ndarray, target: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(len(features) - look_back):
        X.append(features[i:i+look_back])
        y.append(target[i+look_back])
    return np.array(X), np.array(y)

def build_model(look_back: int, n_features: int, config: dict):
    model = Sequential([
        LSTM(config['units'], input_shape=(look_back, n_features)),
        Dropout(config['dropout']),
        Dense(1)
    ])
    model.compile(optimizer=config['optimizer'], loss=config['loss'])
    return model

def print_metrics_table(metrics, dataset_name="Train"):
    table = [
        ["MAE", metrics['mae']],
        ["MSE", metrics['mse']],
        ["RMSE", metrics['rmse']],
        ["MAPE (%)", metrics['mape']],
        ["R2", metrics['r2']]
    ]
    txt = f"\n{dataset_name} Metrics:\n"
    if tabulate:
        txt += tabulate(table, headers=["Metric", "Value"], floatfmt=".4f")
    else:
        txt += "\n".join(f"{metric}: {value:.4f}" for metric, value in table)       
    logger.info(txt)     

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "oil_production_features.csv",
    look_back: int = typer.Option(3, help="Time steps to look back"),
    config_path: Path = MODELS_DIR / "lstm_config.json"
):
    """Train LSTM for oil production forecasting."""
    logger.info(f"Loading training data from {features_path}")
    df = pd.read_csv(features_path, parse_dates=["period"]).sort_values("period")
    X = df.drop(columns=["period"]).values  # All features + target, target is last column

    # Split into train/test
    split_idx = int(len(df) * 0.8)
    train, test = X[:split_idx], X[split_idx:]

    # Split features and target
    X_train_features = train[:, :-1]
    y_train_target = train[:, -1].reshape(-1, 1)

    scaler_X = MinMaxScaler().fit(X_train_features)
    scaler_y = MinMaxScaler().fit(y_train_target)

    X_train_scaled = scaler_X.transform(X_train_features)
    y_train_scaled = scaler_y.transform(y_train_target)

    trainX, trainY = create_sequences(X_train_scaled, y_train_scaled, look_back)
    n_features = X_train_scaled.shape[1]

    config = {
        "units": 64,
        "dropout": 0.2,
        "optimizer": "adam",
        "loss": "mean_squared_error",
        "epochs": 100,
        "batch_size": 32,
        "look_back": look_back,
        "features": n_features
    }

    model = build_model(look_back, n_features, config)

    logger.info("Starting model training...")
    history = model.fit(
        trainX, trainY,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(str(MODELS_DIR / "lstm_model.keras"), save_best_only=True)
        ],
        verbose=1
    )

    # Save scalers and config
    joblib.dump(scaler_X, MODELS_DIR / "scaler_X.pkl")
    joblib.dump(scaler_y, MODELS_DIR / "scaler_y.pkl")
    with open(config_path, "w") as f:
        json.dump(config, f)
    logger.success(f"Training complete. Model and assets saved to {MODELS_DIR}")

    # --- METRICS: Train ---
    train_pred_scaled = model.predict(trainX)
    train_pred = scaler_y.inverse_transform(train_pred_scaled)
    train_true = scaler_y.inverse_transform(trainY)

    mae = mean_absolute_error(train_true, train_pred)
    mse = mean_squared_error(train_true, train_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((train_true - train_pred) / train_true)) * 100
    r2 = r2_score(train_true, train_pred)

    metrics_train = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r2": r2
    }
    print_metrics_table(metrics_train, "Train")

    # --- METRICS: Test ---
    # Prepare test set
    X_test_features = test[:, :-1]
    y_test_target = test[:, -1].reshape(-1, 1)
    X_test_scaled = scaler_X.transform(X_test_features)
    y_test_scaled = scaler_y.transform(y_test_target)
    # Align test for sequences
    def create_sequences_for_test(features, target, look_back):
        X, y = [], []
        for i in range(len(features) - look_back):
            X.append(features[i:i+look_back])
            y.append(target[i+look_back])
        return np.array(X), np.array(y)

    testX, testY = create_sequences_for_test(X_test_scaled, y_test_scaled, look_back)
    test_pred_scaled = model.predict(testX)
    test_pred = scaler_y.inverse_transform(test_pred_scaled)
    test_true = scaler_y.inverse_transform(testY)

    mae = mean_absolute_error(test_true, test_pred)
    mse = mean_squared_error(test_true, test_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test_true - test_pred) / test_true)) * 100
    r2 = r2_score(test_true, test_pred)

    metrics_test = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r2": r2
    }
    print_metrics_table(metrics_test, "Test")

if __name__ == "__main__":
    app()
