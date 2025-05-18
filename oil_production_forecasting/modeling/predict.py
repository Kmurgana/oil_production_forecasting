#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: predict.py
Description: LSTM model inference for oil production forecasting.
Project: Oil Production Forecasting
Author: Kevin Murgana 
Created: 2025-05-17
Version: 1.0.0
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import typer
import sys
import subprocess
from loguru import logger
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from oil_production_forecasting.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

class LSTMPredictor:
    def __init__(
        self,
        model_path: Path = MODELS_DIR / "lstm_model.keras",
        x_scaler_path: Path = MODELS_DIR / "scaler_X.pkl",
        y_scaler_path: Path = MODELS_DIR / "scaler_y.pkl"
    ):
        self.model_path = model_path
        self.x_scaler_path = x_scaler_path
        self.y_scaler_path = y_scaler_path
        
        # Auto-train if model not found
        if not self._check_model_exists():
            logger.warning("Model not found. Initializing training process...")
            self._run_training()
            
        self._verify_assets()
        self._load_assets()
        self.sequence_length = self.model.input_shape[1]

    def _check_model_exists(self):
        return self.model_path.exists()

    def _verify_assets(self):
        missing = []
        for path in [self.model_path, self.x_scaler_path, self.y_scaler_path]:
            if not path.exists():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                f"Missing critical assets after training: {', '.join(missing)}"
            )

    def _run_training(self):
        """Execute training script with dependency checks"""
        try:
            # First ensure python-dotenv exists
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "python-dotenv"],
                check=True,
                capture_output=True
            )
            
            # Now run training
            result = subprocess.run(
                [sys.executable, "-m", "oil_production_forecasting.modeling.train"],
                check=True,
                capture_output=False,
                text=True
            )
            logger.success("Training completed successfully")
        except subprocess.CalledProcessError as e:
            logger.critical(f"Training failed:\n{e.stderr}")
            if "No module named 'dotenv'" in e.stderr:
                logger.error("Required package missing. Try manual install:")
                logger.error("pip install python-dotenv")
            raise RuntimeError("Training failed") from e

    def _load_assets(self):
        """Load model and scalers after verification"""
        logger.info("Loading prediction assets...")
        self.model = load_model(self.model_path)
        self.scaler_X = joblib.load(self.x_scaler_path)
        self.scaler_y = joblib.load(self.y_scaler_path)

    def predict(self, X_df: pd.DataFrame) -> float:
        """Make prediction with input validation"""
        if len(X_df) != self.sequence_length:
            raise ValueError(
                f"Input requires exactly {self.sequence_length} timesteps. "
                f"Received {len(X_df)}"
            )
            
        scaled_data = self.scaler_X.transform(X_df)
        reshaped_data = scaled_data.reshape(1, self.sequence_length, -1)
        prediction = self.model.predict(reshaped_data)
        return self.scaler_y.inverse_transform(prediction).flatten()[0]
    
    def predict_sequence(self, X_seq):
        """
        X_seq: shape (1, 3, n_features) - unscaled features in the same order as training.
        Returns: model prediction in original units (inverse-transformed)
        """
        # 1. Reshape to (n_samples * seq_len, n_features) for scaling, then reshape back
        n_samples, look_back, n_features = X_seq.shape
        X_flat = X_seq.reshape(-1, n_features)
        X_scaled = self.scaler_X.transform(X_flat)
        X_scaled_seq = X_scaled.reshape(n_samples, look_back, n_features)

        # 2. Model prediction
        y_pred_scaled = self.model.predict(X_scaled_seq)
        # 3. Inverse transform the prediction to original units
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred.ravel()  # flatten to 1D array for easy use
    
@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv"
):
    """CLI interface for model predictions"""
    try:
        predictor = LSTMPredictor()
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise typer.Exit(code=1) from e

    logger.info(f"Processing input from {features_path}")
    df = pd.read_csv(features_path, parse_dates=["period"])
    
    try:
        # Select only the required sequence length
        input_data = df.drop(columns=["period", "target"]).tail(predictor.sequence_length)
        prediction = predictor.predict(input_data)
    except ValueError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise typer.Exit(code=1) from e

    # Save results with timestamps
    results = pd.DataFrame({
        "timestamp": pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_value": [prediction],
        "input_sequence_length": [predictor.sequence_length]
    })
    results.to_csv(predictions_path, index=False)
    logger.success(f"Prediction saved to {predictions_path}")

if __name__ == "__main__":
    app()
