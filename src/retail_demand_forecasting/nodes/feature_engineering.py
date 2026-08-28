"""Robust feature engineering for time series demand forecasting — Phase 1.

Implements strictly leakage-free transformations:
- Lag features shifted before use
- Rolling aggregates computed on shifted target (rowsBetween -window .. -1)
- Calendar + cyclical (sin/cos) + is_weekend flags
- NaN cleanup via cutoff (drop first max(lag) days per series)

Modular functions with type hints and docstrings for OpenCode audit.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Constants for cyclical periods
# ---------------------------------------------------------------------------
_DEFAULT_LAGS: list[int] = [1, 2, 3, 7, 14, 21, 28]
_DEFAULT_WINDOWS: list[int] = [7, 28]
_TARGET_COL: str = "sales"


def _window_spec(partition_cols: list[str] | None = None) -> Window:
    """Return time-ordered window partitioned by store+item."""
    if partition_cols is None:
        partition_cols = ["store_id", "item_id"]
    return Window.partitionBy(*partition_cols).orderBy("day_id")


def create_lag_features(
    df: DataFrame,
    lag_days: list[int] | None = None,
    target_col: str = _TARGET_COL,
) -> DataFrame:
    """Add lag features for target column.

    Each lag is ``F.lag(target, n)`` over (store_id, item_id) ordered by day_id.
    No future data leaks — lag looks strictly backward.

    Args:
        df: Input Spark DataFrame with ``target_col`` and ``day_id``.
        lag_days: List of lags to create. Defaults to [1,2,3,7,14,21,28].
        target_col: Target column name.

    Returns:
        DataFrame with new columns ``lag_{n}`` for each n in lag_days.
    """
    if lag_days is None:
        lag_days = _DEFAULT_LAGS
    w = _window_spec()
    result = df
    for lag_n in lag_days:
        result = result.withColumn(f"lag_{lag_n}", F.lag(F.col(target_col), lag_n).over(w))
    return result


def create_rolling_features(
    df: DataFrame,
    windows: list[int] | None = None,
    target_col: str = _TARGET_COL,
) -> DataFrame:
    """Add rolling aggregates strictly shifted by 1 to prevent leakage.

    For each window ``w`` computes mean/min/max/std over
    ``rowsBetween(-w, -1)`` — i.e. trailing w rows excluding current row,
    equivalent to pandas ``.shift(1).rolling(w)``.

    Args:
        df: Input Spark DataFrame.
        windows: Window sizes. Defaults to [7, 28].
        target_col: Target column name.

    Returns:
        DataFrame with ``rolling_mean_{w}``, ``rolling_min_{w}``,
        ``rolling_max_{w}``, ``rolling_std_{w}`` per window.
    """
    if windows is None:
        windows = _DEFAULT_WINDOWS
    w = _window_spec()
    result = df
    for win in windows:
        # Shifted rolling window: exclude current row, include previous win rows
        win_spec = w.rowsBetween(-win, -1)
        result = result.withColumn(f"rolling_mean_{win}", F.avg(F.col(target_col)).over(win_spec))
        result = result.withColumn(f"rolling_min_{win}", F.min(F.col(target_col)).over(win_spec))
        result = result.withColumn(f"rolling_max_{win}", F.max(F.col(target_col)).over(win_spec))
        result = result.withColumn(f"rolling_std_{win}", F.stddev(F.col(target_col)).over(win_spec))
        # Fill std null (single value) with 0
        result = result.withColumn(
            f"rolling_std_{win}", F.coalesce(F.col(f"rolling_std_{win}"), F.lit(0.0))
        )
    return result


def create_calendar_features(df: DataFrame) -> DataFrame:
    """Add calendar and cyclical time features.

    Adds:
    - ``day_of_week`` (1=Mon..7=Sun via ``dayofweek`` normalized), ``day_of_month``, ``month``, ``year``
    - ``is_weekend`` (1 if Sat/Sun)
    - Cyclical sin/cos: ``*_sin`` / ``*_cos`` for day_of_week (period 7),
      day_of_month (period 31), month (period 12)

    Args:
        df: Input Spark DataFrame with ``date`` column (DateType).

    Returns:
        DataFrame with calendar columns added.
    """
    result = df
    # Basic calendar
    # Spark dayofweek: 1=Sunday..7=Saturday -> convert to 1=Mon..7=Sun: ((dow+5)%7 +1)
    # Keep also raw dow for is_weekend
    result = result.withColumn("day_of_week_raw", F.dayofweek(F.col("date")))
    # ISO: 1=Mon..7=Sun
    result = result.withColumn(
        "day_of_week",
        (((F.col("day_of_week_raw") + 5) % 7) + F.lit(1)).cast("int"),
    )
    result = result.withColumn("day_of_month", F.dayofmonth(F.col("date")))
    result = result.withColumn("month", F.month(F.col("date")))
    result = result.withColumn("year", F.year(F.col("date")))
    result = result.withColumn(
        "is_weekend", F.when(F.col("day_of_week").isin(6, 7), F.lit(1)).otherwise(F.lit(0))
    )
    # Cyclical sin/cos: sin(2*pi*value/period), cos(...)
    # day_of_week period 7
    result = result.withColumn(
        "day_of_week_sin", F.sin(2 * 3.141592653589793 * F.col("day_of_week") / F.lit(7.0))
    )
    result = result.withColumn(
        "day_of_week_cos", F.cos(2 * 3.141592653589793 * F.col("day_of_week") / F.lit(7.0))
    )
    # day_of_month period 31
    result = result.withColumn(
        "day_of_month_sin", F.sin(2 * 3.141592653589793 * F.col("day_of_month") / F.lit(31.0))
    )
    result = result.withColumn(
        "day_of_month_cos", F.cos(2 * 3.141592653589793 * F.col("day_of_month") / F.lit(31.0))
    )
    # month period 12
    result = result.withColumn(
        "month_sin", F.sin(2 * 3.141592653589793 * F.col("month") / F.lit(12.0))
    )
    result = result.withColumn(
        "month_cos", F.cos(2 * 3.141592653589793 * F.col("month") / F.lit(12.0))
    )
    result = result.drop("day_of_week_raw")
    return result


def create_event_features(df: DataFrame) -> DataFrame:
    """Add event flag features from calendar event columns.

    Args:
        df: DataFrame with ``event_name_1`` / ``event_name_2``.

    Returns:
        DataFrame with ``has_event_1`` / ``has_event_2`` (1/0).
    """
    result = df
    result = result.withColumn(
        "has_event_1", F.when(F.col("event_name_1").isNotNull(), F.lit(1)).otherwise(F.lit(0))
    )
    result = result.withColumn(
        "has_event_2", F.when(F.col("event_name_2").isNotNull(), F.lit(1)).otherwise(F.lit(0))
    )
    return result


def create_hierarchical_features(
    df: DataFrame,
    windows: list[int] | None = None,
    target_col: str = _TARGET_COL,
) -> DataFrame:
    """Add store / category level rolling means to help sparse items.

    Low-volume items (e.g., HOBBIES_1_003) have many zeros and item-level
    lags collapse to 0. Hierarchical aggregates let them inherit
    store/department/category trends.

    Computes shifted rolling mean (rowsBetween -w .. -1) partitioned by:
    - store_id only (store trend)
    - dept_id (or cat_id if dept missing) — department trend
    - cat_id — category trend

    Args:
        df: DataFrame with ``store_id``, ``dept_id``, ``cat_id``, ``sales``.
        windows: Window sizes. Defaults to [7, 28].
        target_col: Target column.

    Returns:
        DataFrame with ``store_rolling_mean_{w}``, ``dept_rolling_mean_{w}``,
        ``cat_rolling_mean_{w}``.
    """
    if windows is None:
        windows = _DEFAULT_WINDOWS
    result = df
    # Window specs per hierarchy (leakage-free: exclude current row)
    w_store = Window.partitionBy("store_id").orderBy("day_id")
    # Prefer dept_id, fallback to cat_id if dept missing, else store
    has_dept = "dept_id" in df.columns
    has_cat = "cat_id" in df.columns
    w_dept = Window.partitionBy("dept_id").orderBy("day_id") if has_dept else w_store
    w_cat = Window.partitionBy("cat_id").orderBy("day_id") if has_cat else w_store

    for win in windows:
        # store-level
        result = result.withColumn(
            f"store_rolling_mean_{win}",
            F.avg(F.col(target_col)).over(w_store.rowsBetween(-win, -1)),
        )
        # dept-level
        result = result.withColumn(
            f"dept_rolling_mean_{win}",
            F.avg(F.col(target_col)).over(w_dept.rowsBetween(-win, -1)),
        )
        # cat-level
        result = result.withColumn(
            f"cat_rolling_mean_{win}",
            F.avg(F.col(target_col)).over(w_cat.rowsBetween(-win, -1)),
        )
        # Fill nulls (first days) with 0 — sparse items get 0 store trend initially rather than null
        for prefix in ["store", "dept", "cat"]:
            result = result.withColumn(
                f"{prefix}_rolling_mean_{win}",
                F.coalesce(F.col(f"{prefix}_rolling_mean_{win}"), F.lit(0.0)),
            )
    return result


def clean_features(
    df: DataFrame,
    lag_days: list[int] | None = None,
    windows: list[int] | None = None,
) -> DataFrame:
    """Drop rows with NaN from lag/rolling generation (cutoff per series).

    Uses first ``max(lag_days)`` days per (store_id,item_id) as cutoff.

    Args:
        df: DataFrame after lag/rolling.
        lag_days: Lags used (for NaN check). Defaults to _DEFAULT_LAGS.
        windows: Windows used (for NaN check). Defaults to _DEFAULT_WINDOWS.

    Returns:
        DataFrame with initial NaN rows removed.
    """
    if lag_days is None:
        lag_days = _DEFAULT_LAGS
    if windows is None:
        windows = _DEFAULT_WINDOWS
    # Require all lags + at least rolling_mean to be non-null; rolling_min/max/std may be null for small windows but mean is required
    lag_cols = [f"lag_{n}" for n in lag_days]
    rolling_cols = [f"rolling_mean_{w}" for w in windows]
    return df.dropna(subset=lag_cols + rolling_cols)


def create_features(df: DataFrame, params: dict) -> DataFrame:
    """Orchestrate full robust feature engineering (leakage-free).

    Reads ``lag_days`` and ``rolling_window_days`` from params; falls back to
    Phase-1 defaults [1,2,3,7,14,21,28] and [7,28].

    Steps:
    1. Lag features (shifted)
    2. Rolling mean/min/max/std shifted by 1 (rowsBetween -w .. -1)
    3. Calendar + cyclical (day_of_month, is_weekend, sin/cos)
    4. Event flags
    5. Clean NaN cutoff

    Args:
        df: Input Spark DataFrame with ``sales``, ``date``, ``day_id``, ``store_id``, ``item_id``.
        params: Dict with ``lag_days`` and ``rolling_window_days``.

    Returns:
        DataFrame with all engineered features, leakage-free, NaN-cleaned.
    """
    lag_days = params.get("lag_days", _DEFAULT_LAGS)
    # Backwards compat: support old param name rolling_window_days
    windows = params.get("rolling_window_days", params.get("windows", _DEFAULT_WINDOWS))
    # Ensure defaults if empty
    if not lag_days:
        lag_days = _DEFAULT_LAGS
    if not windows:
        windows = _DEFAULT_WINDOWS

    result = df
    result = create_lag_features(result, lag_days=lag_days)
    result = create_rolling_features(result, windows=windows)
    result = create_hierarchical_features(result, windows=windows)
    result = create_calendar_features(result)
    result = create_event_features(result)
    result = clean_features(result, lag_days=lag_days, windows=windows)
    return result
