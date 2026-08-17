"""Unit tests for retail_demand_forecasting nodes."""

import os
import tempfile

import pandas as pd
import pytest

from retail_demand_forecasting.nodes.constants import FEATURE_COLS, TARGET_COL
from retail_demand_forecasting.nodes.data_science import train_model


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Return a small synthetic feature DataFrame for testing."""
    rng = pd.date_range("2016-01-01", periods=100, freq="D")
    rows = []
    for i, dt in enumerate(rng):
        rows.append(
            {
                "day_id": i + 1,
                "date": dt,
                "item_id": "FOODS_1_001",
                "store_id": "CA_1",
                "sales": i % 10,
                "lag_7": (i - 7) % 10 if i >= 7 else 0,
                "lag_28": (i - 28) % 10 if i >= 28 else 0,
                "rolling_mean_7": 4.5,
                "rolling_mean_28": 4.5,
                "day_of_week": dt.dayofweek + 1,
                "month": dt.month,
                "year": dt.year,
                "snap_CA": 1 if i % 3 == 0 else 0,
                "snap_TX": 0,
                "snap_WI": 0,
                "has_event_1": 1 if i % 10 == 0 else 0,
                "has_event_2": 0,
                "sell_price": 1.25,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _patch_mlflow_uri(tmp_path):
    """Point MLflow to a temp SQLite DB so tests don't touch production data."""
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmp_path / 'test_mlflow.db'}"
    yield
    os.environ.pop("MLFLOW_TRACKING_URI", None)


class TestConstants:
    def test_feature_cols_not_empty(self):
        assert len(FEATURE_COLS) > 0

    def test_target_col_is_sales(self):
        assert TARGET_COL == "sales"

    def test_all_features_are_strings(self):
        assert all(isinstance(c, str) for c in FEATURE_COLS)


class TestTrainModel:
    def test_returns_expected_keys(self, sample_features):
        params = {
            "model_params": {"max_depth": 3, "num_trees": 10},
            "target_col": "sales",
            "test_size": 0.2,
            "random_state": 42,
        }
        result = train_model(sample_features, params)
        assert "mae" in result
        assert "rmse" in result
        assert "mape" in result
        assert "features" in result
        assert "train_rows" in result
        assert "test_rows" in result

    def test_metrics_are_non_negative(self, sample_features):
        params = {
            "model_params": {"max_depth": 3, "num_trees": 10},
            "target_col": "sales",
            "test_size": 0.2,
            "random_state": 42,
        }
        result = train_model(sample_features, params)
        assert result["mae"] >= 0
        assert result["rmse"] >= 0
        assert result["mape"] >= 0

    def test_train_test_split_sizes(self, sample_features):
        params = {
            "model_params": {"max_depth": 3, "num_trees": 10},
            "target_col": "sales",
            "test_size": 0.3,
            "random_state": 42,
        }
        result = train_model(sample_features, params)
        total = result["train_rows"] + result["test_rows"]
        assert total == len(sample_features)
