"""Data science node — model training and evaluation with MLflow tracking."""

import json
import logging
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from .constants import FEATURE_COLS, TARGET_COL, MLFLOW_TRACKING_URI

log = logging.getLogger(__name__)


def train_model(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Train a RandomForest regressor and log metrics/artifacts to MLflow.

    Args:
        df: Feature matrix as a pandas DataFrame.
        params: Parameter dict expected to contain ``model_params``,
            ``target_col``, ``test_size``, and ``random_state``.

    Returns:
        Dictionary of evaluation metrics and training metadata.
    """
    model_params = params.get("model_params", {})
    target_col = params.get("target_col", TARGET_COL)
    max_depth = model_params.get("max_depth", 5)
    n_estimators = model_params.get("num_trees", 50)
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    available = [c for c in FEATURE_COLS if c in df.columns]
    log.info("Using features: %s", available)

    X = df[available].fillna(0)
    y = df[target_col].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    log.info("Train: %d rows, Test: %d rows", len(X_train), len(X_test))

    model = RandomForestRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100)

    log.info("MAE: %.4f, RMSE: %.4f, MAPE: %.2f%%", mae, rmse, mape)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name="retail_demand_forecast"):
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("features", json.dumps(available))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mape", mape)
        mlflow.sklearn.log_model(model, "model")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": available,
    }
