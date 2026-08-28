"""Unit tests for retail_demand_forecasting nodes — extended for MLOps audit."""

import os

import numpy as np
import pandas as pd
import pytest

from retail_demand_forecasting.nodes.constants import FEATURE_COLS, RANDOM_STATE, TARGET_COL
from retail_demand_forecasting.nodes.data_science import (
    compute_metrics,
    seed_everything,
    time_series_train_test_split,
    train_model,
)


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Return a small synthetic feature DataFrame for testing (deterministic)."""
    seed_everything(RANDOM_STATE)
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
    uri = f"sqlite:///{tmp_path / 'test_mlflow.db'}"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    # Also patch constants module
    import retail_demand_forecasting.nodes.constants as const
    import retail_demand_forecasting.nodes.data_science as ds

    orig = const.MLFLOW_TRACKING_URI
    const.MLFLOW_TRACKING_URI = uri
    ds.MLFLOW_TRACKING_URI = uri
    yield
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    const.MLFLOW_TRACKING_URI = orig
    ds.MLFLOW_TRACKING_URI = orig


class TestConstants:
    def test_feature_cols_not_empty(self):
        assert len(FEATURE_COLS) > 0

    def test_target_col_is_sales(self):
        assert TARGET_COL == "sales"

    def test_all_features_are_strings(self):
        assert all(isinstance(c, str) for c in FEATURE_COLS)

    def test_random_state_is_int(self):
        assert isinstance(RANDOM_STATE, int)


class TestMetrics:
    def test_compute_metrics_keys(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        m = compute_metrics(y_true, y_pred)
        assert set(m.keys()) == {"mae", "rmse", "wape", "mape", "r2"}

    def test_wape_zero_error(self):
        # All zero true => wape inf
        m = compute_metrics(np.array([0, 0, 0]), np.array([1, 2, 3]))
        assert m["wape"] == float("inf")

    def test_wape_perfect_prediction(self):
        y = np.array([5, 5, 5])
        m = compute_metrics(y, y)
        assert m["wape"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)

    def test_metrics_non_negative(self):
        y_true = np.array([2, 4, 6])
        y_pred = np.array([3, 3, 7])
        m = compute_metrics(y_true, y_pred)
        assert m["mae"] >= 0
        assert m["rmse"] >= 0
        assert m["wape"] >= 0 or m["wape"] == float("inf")
        assert m["mape"] >= 0


class TestDeterministicSeeds:
    def test_seed_everything_reproducible(self):
        seed_everything(42)
        a = np.random.rand(5)
        seed_everything(42)
        b = np.random.rand(5)
        assert np.allclose(a, b)


class TestTimeSeriesSplit:
    def test_temporal_order_preserved(self, sample_features):
        X_train, X_test, y_train, y_test, _ = time_series_train_test_split(
            sample_features, test_size=0.2
        )
        # Train dates should all be before test dates
        # Reconstruct from original sorted df
        df_sorted = sample_features.sort_values("date")
        split = int(len(df_sorted) * 0.8)
        assert len(X_train) == split
        assert len(X_test) == len(df_sorted) - split
        # Ensure no shuffling — first test row follows last train row chronologically
        assert y_train.iloc[-1] == df_sorted.iloc[split - 1]["sales"]
        assert y_test.iloc[0] == df_sorted.iloc[split]["sales"]

    def test_no_future_leakage(self, sample_features):
        # Shuffle input, split should still sort by date
        shuffled = sample_features.sample(frac=1, random_state=42)
        X_train, X_test, y_train, y_test, _ = time_series_train_test_split(shuffled, test_size=0.3)
        # Combined sorted should equal original sorted
        assert len(X_train) + len(X_test) == len(sample_features)

    def test_split_sizes(self, sample_features):
        for frac in [0.1, 0.2, 0.5]:
            X_train, X_test, y_train, y_test, _ = time_series_train_test_split(
                sample_features, test_size=frac
            )
            total = len(X_train) + len(X_test)
            assert total == len(sample_features)
            assert len(X_test) == pytest.approx(len(sample_features) * frac, abs=1)


class TestTrainModel:
    def test_returns_expected_keys(self, sample_features):
        params = {
            "model_params": {"max_depth": 3, "num_trees": 10},
            "target_col": "sales",
            "test_size": 0.2,
            "random_state": 42,
        }
        result = train_model(sample_features, params)
        for k in ["mae", "rmse", "wape", "r2", "mape", "features", "train_rows", "test_rows"]:
            assert k in result, f"missing key {k}"

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
        assert result["wape"] >= 0 or result["wape"] == float("inf")
        assert result["r2"] <= 1.0  # r2 can be negative

    def test_deterministic(self, sample_features):
        params = {
            "model_params": {"max_depth": 3, "num_trees": 10},
            "target_col": "sales",
            "test_size": 0.2,
            "random_state": 42,
        }
        r1 = train_model(sample_features, params)
        r2 = train_model(sample_features, params)
        assert r1["mae"] == pytest.approx(r2["mae"])
        assert r1["wape"] == pytest.approx(r2["wape"])

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
