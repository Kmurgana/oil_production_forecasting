#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: config.py
Description: Configuration file for oil production forecasting project.
Project: Oil Production Forecasting
Author: Kevin Murgana 
Created: 2025-05-17
Version: 1.0.0
"""

from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/processed/oil_production_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../models/")
RESULTS_DIR = os.path.join(BASE_DIR, "../results/")
BEST_MODEL_FLAG = os.path.join(MODEL_DIR, "best_model_flag.txt")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
