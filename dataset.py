"""Load and split the car fuel-efficiency table used by this homework."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "car_fuel_efficiency_2026.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
)

FEATURE_COLUMNS = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
]
TARGET = "fuel_efficiency_mpg"
COLUMNS = FEATURE_COLUMNS + [TARGET]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Return only the columns used to predict fuel efficiency."""
    frame = pd.read_csv(path)
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return frame[COLUMNS].copy()


def train_val_test_split(df: pd.DataFrame, seed: int = 42):
    """60/20/20 split after a seeded shuffle, matching the lecture exactly."""
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]
    return df_train, df_val, df_test


def prepare_X(df: pd.DataFrame, fill_value: float) -> np.ndarray:
    """Feature matrix with missing entries replaced by one fill value."""
    features = df[FEATURE_COLUMNS].copy()
    features = features.fillna(fill_value)
    return features.to_numpy(dtype=float)


def extract_y(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET].to_numpy(dtype=float)
