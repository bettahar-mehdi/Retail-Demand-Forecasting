from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql.functions import (
    col,
    lag,
    avg as _avg,
    dayofweek,
    month,
    year,
    when,
)


def create_features(df: DataFrame, params: dict) -> DataFrame:
    """Create lag and rolling window features using PySpark Window functions.

    Reads lag_days and rolling_window_days from parameters.
    Partitions by (store_id, item_id) and orders by date.
    """
    lag_days = params.get("lag_days", [7, 28])
    rolling_window_days = params.get("rolling_window_days", [7, 28])

    # Window: partition by store+item, ordered by day_id
    w = Window.partitionBy("store_id", "item_id").orderBy("day_id")

    result = df

    # --- Lag features ---
    for lag_n in lag_days:
        result = result.withColumn(
            f"lag_{lag_n}",
            lag("sales", lag_n).over(w),
        )

    # --- Rolling mean features ---
    for window_n in rolling_window_days:
        result = result.withColumn(
            f"rolling_mean_{window_n}",
            _avg("sales").over(w.rowsBetween(-window_n, -1)),
        )

    # --- Calendar features ---
    result = result.withColumn("day_of_week", dayofweek(col("date")))
    result = result.withColumn("month", month(col("date")))
    result = result.withColumn("year", year(col("date")))

    # --- Event flags ---
    result = result.withColumn(
        "has_event_1",
        when(col("event_name_1").isNotNull(), 1).otherwise(0),
    )
    result = result.withColumn(
        "has_event_2",
        when(col("event_name_2").isNotNull(), 1).otherwise(0),
    )

    # Drop rows where lag features are null (first N days of each series)
    result = result.dropna(
        subset=[f"lag_{l}" for l in lag_days]
        + [f"rolling_mean_{w}" for w in rolling_window_days]
    )

    return result
