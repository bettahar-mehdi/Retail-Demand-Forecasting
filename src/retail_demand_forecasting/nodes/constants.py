"""Shared constants for feature columns and defaults."""

import os

FEATURE_COLS: list[str] = [
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "month",
    "year",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "has_event_1",
    "has_event_2",
    "sell_price",
]

TARGET_COL: str = "sales"

MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db",
)
