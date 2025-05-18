
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: dataset.py
Description: Data processing and feature engineering for oil production forecasting.
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
from oil_production_forecasting.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def load_feature_dataset() -> pd.DataFrame:
    """Load processed dataset with all engineered features."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "oil_production_features.csv", parse_dates=["period"])
    df = df.sort_values("period")
    return df

def split_dataset(df: pd.DataFrame, test_ratio=0.2):
    """Split into train and test."""
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "oil_production_padd3_1960_present.csv",
    output_path: Path = PROCESSED_DATA_DIR / "oil_production_features.csv",
):
    """
    Load raw dataset, process it, and save the processed dataset.
    Args:
        input_path (Path): Path to the raw dataset.
        output_path (Path): Path to save the processed dataset.
    
    Returns:
        None
    """
    logger.info("Loading raw dataset...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {df.shape[0]} rows.")

    # Example basic transformation (you can replace this with real processing)
    df['period'] = pd.to_datetime(df['period'])
    df = df.sort_values('period')

    # Save processed data
    df.to_csv(output_path, index=False)
    logger.success(f"Saved processed dataset to {output_path}")


if __name__ == "__main__":
    app()
