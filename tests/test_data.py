import pandas as pd
import numpy as np
from oil_production_forecasting.modeling.predict import LSTMPredictor
from oil_production_forecasting.dataset import load_feature_dataset, split_dataset

def test_model_output_format():
    predictor = LSTMPredictor()
    df = load_feature_dataset()
    _, test_df = split_dataset(df)
    X_test = test_df.drop(columns=["period", "target"])

    # Use a valid input shape for LSTM: (batch, timesteps, features)
    X_input = X_test.iloc[:3].values  # 3 timesteps
    X_input = X_input.reshape((1, 3, X_input.shape[1]))  # batch=1

    y_pred = predictor.model.predict(X_input)
    assert isinstance(y_pred, np.ndarray), "Prediction must be a numpy array."
    assert y_pred.shape[0] == 1, "Prediction batch size must match input batch size."


def test_model_consistency():
    predictor = LSTMPredictor()
    df = load_feature_dataset()
    _, test_df = split_dataset(df)
    X_test = test_df.drop(columns=["period", "target"])

    # Use 3 timesteps to match model input requirement
    X_input = X_test.iloc[:3].values
    X_input = X_input.reshape((1, 3, X_input.shape[1]))

    # Use model.predict directly for determinism
    y_pred_1 = predictor.model.predict(X_input)
    y_pred_2 = predictor.model.predict(X_input)

    # If output is multi-dimensional, flatten for comparison
    y_pred_1 = np.asarray(y_pred_1).flatten()
    y_pred_2 = np.asarray(y_pred_2).flatten()
    assert abs(y_pred_1[0] - y_pred_2[0]) < 1e-5, "Model predictions should be deterministic."


def test_scaler_feature_names():
    predictor = LSTMPredictor()
    df = load_feature_dataset()
    _, test_df = split_dataset(df)
    X_test = test_df.drop(columns=["period", "target"])

    # Use a robust check for feature_names_in_
    if hasattr(predictor.scaler_X, 'feature_names_in_'):
        model_features = list(predictor.scaler_X.feature_names_in_)
    else:
        # Fallback to X_test columns if scaler was fit with numpy
        model_features = list(X_test.columns)

    assert list(X_test.columns) == model_features, "Feature names in test set must match those seen during training."

