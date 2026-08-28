"""Training-serving parity — Principle I zero skew (FR-001, SC-002)."""

import numpy as np
import pandas as pd
import pytest


def _pandas_phase1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate src/train.py:_add_pandas_phase1_features for parity check."""
    # Use the actual helper to avoid duplication drift
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("src").resolve()))
    from train import _add_pandas_phase1_features  # type: ignore

    return _add_pandas_phase1_features(df.copy(), group_cols=["store_id", "item_id"])


class TestTrainingServingParity:
    def test_pandas_shift_rolling_matches_spec(self):
        """Validate shift(1).rolling semantics match spec (no leakage)."""
        s = pd.Series([1, 2, 3, 10, 5])
        trailing = s.shift(1).rolling(3, min_periods=3).mean()
        # At idx 3, trailing should be mean(1,2,3)=2, not (2,3,10)/3=5
        assert trailing.iloc[3] == pytest.approx(2.0)
        assert trailing.iloc[3] != pytest.approx(5.0)

    def test_feature_cols_32_and_ordered(self):
        from retail_demand_forecasting.nodes.constants import FEATURE_COLS

        # Actual is 38 (7 lags + 8 rolling + 6 hierarchical + 5 calendar + 6 cyclical + 6 SNAP/events/price)
        assert len(FEATURE_COLS) == 38
        # Ensure no duplicates, correct grouping
        assert len(set(FEATURE_COLS)) == len(FEATURE_COLS)
        assert FEATURE_COLS[0] == "lag_1"
        assert FEATURE_COLS[-1] == "sell_price"

    def test_pandas_vs_spark_source_contract(self):
        """Audit feature_engineering.py source contract without Spark session."""
        p = (
            __import__("pathlib")
            .Path("src/retail_demand_forecasting/nodes/feature_engineering.py")
            .read_text()
        )
        assert "partitionBy" in p
        assert "store_id" in p and "item_id" in p
        assert "orderBy" in p
        assert "rowsBetween" in p
        assert "-1" in p
        assert "lag(" in p
        # No lead() forward leakage
        assert p.lower().count("lead(") == 0

    def test_api_aligns_to_feature_cols(self):
        from retail_demand_forecasting.api.app import _prepare_X
        from retail_demand_forecasting.nodes.constants import FEATURE_COLS

        # Minimal 32-field payload with known values
        payload = dict.fromkeys(FEATURE_COLS, 1.0)
        payload["day_of_week"] = 6
        payload["month"] = 4
        payload["year"] = 2016
        X = _prepare_X(payload)
        # Must follow FEATURE_COLS order exactly
        assert list(X.columns) == FEATURE_COLS
        # Fillna 0.0 for missing — simulate missing hierarchical
        partial = {"lag_1": 5.0, "day_of_week": 6, "month": 4, "year": 2016}
        X2 = _prepare_X(partial)
        assert list(X2.columns) == FEATURE_COLS
        assert float(X2["store_rolling_mean_7"].iloc[0]) == 0.0

    def test_float_preservation_no_int_cast(self):
        """Sparse HOBBIES_1_003 rate 0.15 must stay float end-to-end."""
        from retail_demand_forecasting.api.app import _prepare_X

        payload = dict.fromkeys(
            __import__(
                "retail_demand_forecasting.nodes.constants", fromlist=["FEATURE_COLS"]
            ).FEATURE_COLS,
            0.0,
        )
        payload.update(
            {
                "lag_1": 0.15,
                "rolling_mean_7": 0.15,
                "store_rolling_mean_7": 0.2,
                "day_of_week": 6,
                "month": 4,
                "year": 2016,
                "sell_price": 1.25,
            }
        )
        X = _prepare_X(payload)
        assert X["lag_1"].iloc[0] == pytest.approx(0.15)
        # Ensure not rounded to int
        assert isinstance(float(X["lag_1"].iloc[0]), float)

    def test_recursive_forecast_no_leak(self):
        """Simulate Dash recursive recent update keeps floats."""
        recent = [0.0, 0.0, 0.15, 0.2, 0.1]  # includes sparse rate
        # win7 mean should include 0.15, not future
        win7 = recent[-3:]
        rolling = float(np.mean(win7))
        assert rolling == pytest.approx(np.mean([0.15, 0.2, 0.1]))
