"""Load the car fuel-efficiency table used by this homework."""

from pathlib import Path

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
