"""Shared constants for feature columns and defaults — Phase 1 robust."""

import os

# Phase 1 + Hierarchical: Lag [1,2,3,7,14,21,28] + Rolling 7/28 mean/min/max/std + Calendar/cyclical + Hierarchical store/dept/cat
FEATURE_COLS: list[str] = [
    # Lags
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    # Rolling 7
    "rolling_mean_7",
    "rolling_min_7",
    "rolling_max_7",
    "rolling_std_7",
    # Rolling 28
    "rolling_mean_28",
    "rolling_min_28",
    "rolling_max_28",
    "rolling_std_28",
    # Hierarchical store/dept/cat rolling means (helps sparse HOBBIES_1_003)
    "store_rolling_mean_7",
    "store_rolling_mean_28",
    "dept_rolling_mean_7",
    "dept_rolling_mean_28",
    "cat_rolling_mean_7",
    "cat_rolling_mean_28",
    # Calendar
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "is_weekend",
    # Cyclical sin/cos
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_month_sin",
    "day_of_month_cos",
    "month_sin",
    "month_cos",
    # SNAP / events / price
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "has_event_1",
    "has_event_2",
    "sell_price",
]

TARGET_COL: str = "sales"

RANDOM_STATE: int = 42

MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db",
)
