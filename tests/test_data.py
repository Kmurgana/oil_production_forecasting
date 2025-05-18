import pandas as pd
import numpy as np
from oil_production_forecasting.modeling.predict import LSTMPredictor
from oil_production_forecasting.dataset import load_feature_dataset, split_dataset

def test_model_output_format():
    predictor = LSTMPredictor()
    df = load_feature_dataset()
    _, test_df = split_dataset(df)
    X_test = test_df.drop(columns=["period", "target"])

    y_pred = predictor.predict(X_test)

    assert isinstance(y_pred, (list, pd.Series, np.ndarray)), "Prediction must be array-like."
    assert len(y_pred) == len(X_test), "Prediction length must match input length."

def test_model_consistency():
    predictor = LSTMPredictor()
    df = load_feature_dataset()
    _, test_df = split_dataset(df)
    X_test = test_df.drop(columns=["period", "target"])

    y_pred_1 = predictor.predict(X_test.iloc[:1])
    y_pred_2 = predictor.predict(X_test.iloc[:1])

    assert abs(y_pred_1[0] - y_pred_2[0]) < 1e-5, "Model predictions should be deterministic."

def test_scaler_feature_names():
    predictor = LSTMPredictor()
    df = load_feature_dataset()
    _, test_df = split_dataset(df)
    X_test = test_df.drop(columns=["period", "target"])
    model_features = list(predictor.scaler_X.feature_names_in_)

    assert list(X_test.columns) == model_features, "Feature names in test set must match those seen during training."
