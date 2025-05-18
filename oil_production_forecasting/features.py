#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: features.py
Description: Feature engineering for oil production forecasting.
Project: Oil Production Forecasting
Author: Kevin Murgana 
Created: 2025-05-17
Version: 1.0.0
"""

from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer

from oil_production_forecasting.config import PROCESSED_DATA_DIR

app = typer.Typer()

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate time-based, lag, and rolling features.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the dataset with 'period', 'product-name', 'process-name', and 'value' columns.
    
    Returns:
        pd.DataFrame: DataFrame with additional features.
    """
    logger.info("Starting feature engineering...")

    df['period'] = pd.to_datetime(df['period'])
    df = df.sort_values(by='period').reset_index(drop=True)

    # Filter for main product (Crude Oil + Field Production)
    df = df[(df['product-name'] == 'Crude Oil') & 
                (df['process-name'] == 'Field Production')]

    # Create a pivot table
    df = df.pivot_table(index='period', 
                                columns='units', 
                                values='value', 
                                aggfunc='sum')

    # Set period as another column.
    df = df.reset_index()
    print(list(df.columns))

    df['year'] = df['period'].dt.year
    df['month'] = df['period'].dt.month
    df['quarter'] = df['period'].dt.quarter
    df['dayofyear'] = df['period'].dt.dayofyear
    df['weekofyear'] = df['period'].dt.isocalendar().week.astype(int)

    print(list(df.columns))
    # Lag features
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df['MBBL/D'].shift(lag)

    # Rolling features
    df['rolling_mean_3'] = df['MBBL/D'].shift(1).rolling(window=3).mean()
    df['rolling_mean_6'] = df['MBBL/D'].shift(1).rolling(window=6).mean()
    df['rolling_std_3'] = df['MBBL/D'].shift(1).rolling(window=3).std()
    df['rolling_std_6'] = df['MBBL/D'].shift(1).rolling(window=6).std()

    # Target column for supervised learning (t+1 forecast)
    df['target'] = df['MBBL/D'].shift(-1)

    df = df.dropna().reset_index(drop=True)
    logger.success("Feature engineering complete.")
    return df

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "oil_production_features.csv",
):
    """CLI command to generate features from a cleaned dataset."""
    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path, parse_dates=["period"])
    df = df.sort_values("period")

    df_features = generate_features(df)

    df_features.to_csv(output_path, index=False)
    logger.success(f"Saved features to {output_path}")


if __name__ == "__main__":
    app()

