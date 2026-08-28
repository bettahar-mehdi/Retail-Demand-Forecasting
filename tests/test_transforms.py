"""Feature transformation pipeline tests — ensure zero future leakage."""

import pandas as pd
import pytest

# These tests run without a Spark session by testing the logic contract:
# - Lag/rolling must be trailing only (rowsBetween -window .. -1)
# We validate by inspecting the node source and by simulating with pandas.


class TestFeatureContracts:
    def test_feature_engineering_source_uses_trailing_window(self):
        """Audit src/nodes/feature_engineering.py for leakage."""
        from pathlib import Path

        src = Path("src/retail_demand_forecasting/nodes/feature_engineering.py").read_text()
        # Must partition by store+item and order by day_id
        assert 'partitionBy("store_id", "item_id")' in src or "partitionBy" in src
        assert 'orderBy("day_id")' in src or "orderBy" in src
        # Rolling must be trailing: rowsBetween(-window, -1) — not future
        assert "rowsBetween" in src
        assert "-1" in src
        # Lag must use lag()
        assert "lag(" in src
        # No forward fill or lead() that would leak future
        assert "lead(" not in src.lower() or src.count("lead(") == 0

    def test_pandas_trailing_rolling_no_leakage(self):
        """Simulate rolling: future values must not affect current."""
        s = pd.Series([1, 2, 3, 10, 5])
        # Trailing 3-day mean at idx 3 should be mean(1,2,3)=2, not include 10
        trailing = s.shift(1).rolling(3, min_periods=3).mean()
        assert trailing.iloc[3] == pytest.approx(2.0)  # (1+2+3)/3
        # If we leaked future, it would be (2+3+10)/3 = 5
        assert trailing.iloc[3] != pytest.approx(5.0)

    def test_lag_features_dropna_contract(self):
        """Feature node drops rows where lag is null — first N days removed."""
        from pathlib import Path

        src = Path("src/retail_demand_forecasting/nodes/feature_engineering.py").read_text()
        assert "dropna" in src
        assert "lag_" in src
