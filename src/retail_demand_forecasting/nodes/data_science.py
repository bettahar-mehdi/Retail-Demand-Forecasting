"""Data science node — deterministic time-series training & Optuna tuning.

Implements:
- Deterministic seeds across random / numpy / sklearn / lightgbm / xgboost
- Time-series aware split (no random shuffling — preserves temporal order)
- Metrics: WAPE, RMSE, MAE, R2 (and MAPE for backwards compatibility)
- Optuna rolling-window CV objective (spec §3)
- MLflow logging to sqlite:///mlflow.db
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .constants import FEATURE_COLS, MLFLOW_TRACKING_URI, RANDOM_STATE, TARGET_COL

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic seeds
# ---------------------------------------------------------------------------


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Fix seeds across random, numpy, and python hash seed."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Metrics — WAPE is the primary demand-forecasting metric (spec §3)
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # WAPE = sum(|y - yhat|) / sum(|y|) * 100  (robust to zeros)
    denom = np.sum(np.abs(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom != 0 else float("inf")
    # MAPE variant (capped denominator at 1 to avoid div/0)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    return {"mae": mae, "rmse": rmse, "wape": wape, "mape": mape, "r2": r2}


# ---------------------------------------------------------------------------
# Time-series split helper — zero leakage
# ---------------------------------------------------------------------------


def time_series_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    sort_col: str | None = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Temporal split: last test_size fraction is held out as test set.

    No shuffling — preserves time ordering to avoid future leakage.
    If sort_col exists, sorts by it; otherwise preserves input order.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    if not available:
        raise ValueError(
            f"No feature columns found. Expected one of {FEATURE_COLS}, got {list(df.columns)}"
        )
    target = TARGET_COL
    # Sort chronologically if possible
    if sort_col and sort_col in df.columns:
        df_sorted = df.sort_values(sort_col).reset_index(drop=True)
    elif "day_id" in df.columns:
        df_sorted = df.sort_values("day_id").reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    n = len(df_sorted)
    split_idx = int(n * (1 - test_size))
    # Ensure at least 1 row in each split
    split_idx = max(1, min(n - 1, split_idx))

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train = train_df[available].fillna(0)
    X_test = test_df[available].fillna(0)
    y_train = train_df[target].fillna(0)
    y_test = test_df[target].fillna(0)
    return X_train, X_test, y_train, y_test, available


# ---------------------------------------------------------------------------
# Model factory — zero-inflated support (tweedie/poisson)
# ---------------------------------------------------------------------------


def _build_model(model_params: dict[str, Any], random_state: int = RANDOM_STATE):
    """Build regressor per model_type with count-appropriate objective.

    For sparse retail (HOBBIES_1_003 ~90% zeros), MSE regression collapses to 0.
    Tweedie (variance_power 1.1-1.5) or Poisson properly models zero-inflated counts.

    Supported model_type:
    - RandomForest: baseline MSE (kept for compat)
    - LightGBM: objective tweedie/poisson, tweedie_variance_power 1.1-1.5
    - XGBoost: objective reg:tweedie / count:poisson

    Falls back to RandomForest if lightgbm/xgboost not installed.
    """
    model_type = str(model_params.get("model_type", "RandomForest")).lower()
    # Common tweedie power for retail (1.1-1.5, 1.5 is more zero-tolerant)
    tweedie_p = float(model_params.get("tweedie_variance_power", 1.3))
    learning_rate = float(model_params.get("learning_rate", 0.05))
    n_estimators = int(model_params.get("n_estimators", model_params.get("num_trees", 100)))
    max_depth = int(model_params.get("max_depth", 8))
    # LightGBM
    if model_type in ("lightgbm", "lgbm", "lgb"):
        try:
            import lightgbm as lgb

            # Tweedie needs positive label; LightGBM handles it natively
            objective = str(model_params.get("objective", "tweedie")).lower()
            if objective not in ("tweedie", "poisson", "regression"):
                objective = "tweedie"
            return lgb.LGBMRegressor(
                objective=objective,
                tweedie_variance_power=tweedie_p if objective == "tweedie" else 1.5,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=int(model_params.get("num_leaves", 31)),
                min_child_samples=int(model_params.get("min_child_samples", 20)),
                subsample=float(model_params.get("subsample", 0.8)),
                colsample_bytree=float(model_params.get("colsample_bytree", 0.8)),
                reg_alpha=float(model_params.get("reg_alpha", 0.0)),
                reg_lambda=float(model_params.get("reg_lambda", 0.0)),
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        except ImportError:
            log.warning("lightgbm not installed, falling back to RandomForest")
    if model_type in ("xgboost", "xgb", "xg"):
        try:
            import xgboost as xgb

            objective = str(model_params.get("objective", "tweedie")).lower()
            # XGBoost mapping: reg:tweedie or count:poisson
            if objective == "tweedie":
                xgb_objective = "reg:tweedie"
            elif objective == "poisson":
                xgb_objective = "count:poisson"
            else:
                xgb_objective = "reg:squarederror"
            return xgb.XGBRegressor(
                objective=xgb_objective,
                tweedie_variance_power=tweedie_p if "tweedie" in xgb_objective else None,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=float(model_params.get("subsample", 0.8)),
                colsample_bytree=float(model_params.get("colsample_bytree", 0.8)),
                reg_alpha=float(model_params.get("reg_alpha", 0.0)),
                reg_lambda=float(model_params.get("reg_lambda", 0.0)),
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        except ImportError:
            log.warning("xgboost not installed, falling back to RandomForest")
    # Default RandomForest (MSE) — kept for tests, but sparse items will flatten to 0
    return RandomForestRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Core training — deterministic with chronological split
# ---------------------------------------------------------------------------


def train_model(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Train deterministic model (Tweedie/Poisson for sparse) on time-ordered data and log to MLflow.

    Args:
        df: Feature matrix as a pandas DataFrame (must contain FEATURE_COLS + sales + date/day_id).
        params: Dict with keys ``model_params``, ``target_col``, ``test_size``, ``random_state``.
    Returns:
        Dict of metrics + metadata (mae, rmse, wape, r2, mape, train_rows, test_rows, features).
    """
    seed_everything(params.get("random_state", RANDOM_STATE))

    model_params = params.get("model_params", {})
    _target_col = params.get("target_col", TARGET_COL)  # kept for compat
    test_size = params.get("test_size", 0.2)
    if "test_size" in model_params:
        test_size = model_params["test_size"]
    random_state = params.get("random_state", RANDOM_STATE)
    if "random_state" in model_params:
        random_state = model_params["random_state"]

    X_train, X_test, y_train, y_test, available = time_series_train_test_split(
        df, test_size=test_size
    )
    log.info("Features: %s | Train: %d | Test: %d", available, len(X_train), len(X_test))

    model = _build_model(model_params, random_state=random_state)
    # For hierarchical features, ensure fill 0 for sparse prior
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Clip tiny negatives from tweedie/poisson before metrics (keep float, not int)
    y_pred = np.maximum(0.0, np.asarray(y_pred, dtype=float))

    metrics = compute_metrics(y_test, y_pred)
    log.info(
        "MAE: %.4f RMSE: %.4f WAPE: %.2f%% R2: %.4f",
        metrics["mae"],
        metrics["rmse"],
        metrics["wape"],
        metrics["r2"],
    )

    # Persist best model binary for DVC / serving
    _persist_model(model, Path("models/forecast_model.pkl"))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Log to mlflow.db — single run with full metrics
    with mlflow.start_run(run_name="retail_demand_forecast"):
        # Log all model params generically
        for k, v in model_params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                mlflow.log_param(k, str(v))
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("features", json.dumps(available))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("model_type", str(model_params.get("model_type", "RandomForest")))
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        # Log with appropriate flavor
        try:
            kind = str(model_params.get("model_type", "RandomForest")).lower()
            if kind in ("lightgbm", "lgbm", "lgb"):
                mlflow.lightgbm.log_model(model, artifact_path="model")
            elif kind in ("xgboost", "xgb", "xg"):
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
        except TypeError:
            # fallback for older signature
            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception:
                pass

    return {
        **metrics,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": available,
    }


def _persist_model(model: Any, path: Path) -> None:
    """Save model pickle to path (used by DVC stage outs)."""
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("Model saved to %s", path)


# ---------------------------------------------------------------------------
# Optuna tuning with TimeSeriesSplit CV — spec §3
# ---------------------------------------------------------------------------


def optuna_objective(
    trial: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 3,
    base_params: dict[str, Any] | None = None,
) -> float:
    """Optuna objective: minimize mean WAPE across rolling window CV.

    Supports LightGBM tweedie search space (retail sparse) or RandomForest.
    Tweedie variance_power 1.1-1.5 is critical for zero-inflated counts like HOBBIES_1_003.
    """

    seed_everything(RANDOM_STATE)
    base_params = base_params or {}
    model_type = str(base_params.get("model_type", "RandomForest")).lower()

    if model_type in ("lightgbm", "lgbm", "lgb"):
        params = {
            "model_type": "LightGBM",
            "objective": trial.suggest_categorical("objective", ["tweedie", "poisson"]),
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.5),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        # When poisson, variance_power unused
        if params["objective"] == "poisson":
            params.pop("tweedie_variance_power", None)
    elif model_type in ("xgboost", "xgb", "xg"):
        params = {
            "model_type": "XGBoost",
            "objective": trial.suggest_categorical("objective", ["tweedie", "poisson"]),
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.5),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    else:
        # RandomForest fallback
        params = {
            "model_type": "RandomForest",
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    wapes: list[float] = []
    for train_idx, valid_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
        # Fill NaN for hierarchical sparse prior
        X_tr = X_tr.fillna(0)
        X_val = X_val.fillna(0)
        m = _build_model(params, random_state=RANDOM_STATE)
        m.fit(X_tr, y_tr)
        pred = np.maximum(0.0, np.asarray(m.predict(X_val), dtype=float))
        wape = compute_metrics(y_val, pred)["wape"]
        wapes.append(wape)

    return float(np.mean(wapes))


def tune_and_train(
    df: pd.DataFrame,
    params: dict[str, Any],
    n_trials: int = 20,
    cv_splits: int = 3,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Run Optuna tuning with rolling-window CV, retrain best model, log to MLflow.

    Logs each trial as a nested MLflow run and the final best model as the parent run.
    Saves best model to models/forecast_model.pkl and metrics.json.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna is required for tuning. pip install optuna") from exc

    seed_everything(params.get("random_state", RANDOM_STATE))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train_full, X_test, y_train_full, y_test, available = time_series_train_test_split(
        df, test_size=params.get("test_size", 0.2)
    )
    # For CV we use the training portion only
    X_cv = X_train_full
    y_cv = y_train_full

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    best_params: dict[str, Any] = {}
    best_value = float("inf")

    # Pass base model_type to objective so it searches correct space (tweedie for LightGBM etc.)
    base_model_params = params.get("model_params", {})
    with mlflow.start_run(run_name="optuna_tuning"):
        # Use deterministic sampler
        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            lambda trial: optuna_objective(
                trial, X_cv, y_cv, cv_splits=cv_splits, base_params=base_model_params
            ),
            n_trials=n_trials,
            timeout=timeout,
        )
        best_params = study.best_params
        best_value = study.best_value
        # Ensure model_type preserved for rebuild
        if "model_type" not in best_params and "model_type" in base_model_params:
            best_params["model_type"] = base_model_params["model_type"]

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_wape", best_value)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv_splits", cv_splits)
        mlflow.log_param("features", json.dumps(available))

        # Log all trials as metrics for visibility
        for i, t in enumerate(study.trials):
            if t.value is not None:
                mlflow.log_metric(f"trial_{i}_wape", float(t.value))

        # Retrain on full training set with best params and evaluate on holdout test
        X_train_full = X_train_full.fillna(0)
        X_test_f = X_test.fillna(0)
        best_model = _build_model(best_params, random_state=RANDOM_STATE)
        best_model.fit(X_train_full, y_train_full)
        y_pred = np.maximum(0.0, np.asarray(best_model.predict(X_test_f), dtype=float))
        metrics = compute_metrics(y_test, y_pred)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_metric("holdout_wape", metrics["wape"])
        # Log with correct flavor
        try:
            kind = str(
                best_params.get("model_type", base_model_params.get("model_type", "RandomForest"))
            ).lower()
            if kind in ("lightgbm", "lgbm", "lgb"):
                mlflow.lightgbm.log_model(best_model, artifact_path="model_best")
            elif kind in ("xgboost", "xgb", "xg"):
                mlflow.xgboost.log_model(best_model, artifact_path="model_best")
            else:
                mlflow.sklearn.log_model(best_model, artifact_path="model_best")
        except TypeError:
            try:
                mlflow.sklearn.log_model(best_model, "model_best")
            except Exception:
                pass

        _persist_model(best_model, Path("models/forecast_model.pkl"))

        # Save metrics.json for DVC metrics
        metrics_path = Path("metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(
                {**metrics, "best_params": best_params, "best_cv_wape": best_value}, f, indent=2
            )

        # Save params for reproducibility
        log.info(
            "Best params: %s | Best CV WAPE: %.2f | Holdout: %s", best_params, best_value, metrics
        )

    return {
        **metrics,
        "best_params": best_params,
        "best_cv_wape": best_value,
        "train_rows": len(X_train_full),
        "test_rows": len(X_test),
        "features": available,
    }
