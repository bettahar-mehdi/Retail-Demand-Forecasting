from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr


def unpivot_sales(
    sales_train_raw: DataFrame,
    calendar_raw: DataFrame,
    sell_prices_raw: DataFrame,
) -> DataFrame:
    """Melt wide sales data into long format and join with calendar + prices.

    The sales_train_validation.csv has columns d_1..d_1913.
    We use selectExpr + stack to unpivot them into (day_id, sales),
    then join with calendar (for date mapping) and sell_prices (for price).
    """
    # --- 1. Unpivot d_1..d_1913 using stack ---
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [f"d_{i}" for i in range(1, 1914)]

    stack_expr = ", ".join([f"'{c}', `{c}`" for c in day_cols])

    select_expr = id_cols + [f"stack(1913, {stack_expr}) as (day_id, sales)"]

    melted = sales_train_raw.selectExpr(*select_expr)

    # Cast day_id from string to integer ("d_1" -> 1)
    melted = melted.withColumn(
        "day_id",
        expr("cast(regexp_replace(day_id, 'd_', '') as int)"),
    )

    # --- 2. Prepare calendar: map day_id (d_1 -> 1) to date ---
    calendar = (
        calendar_raw.select(
            col("d").alias("day_id_str"),
            col("date"),
            col("wm_yr_wk"),
            col("event_name_1"),
            col("event_type_1"),
            col("event_name_2"),
            col("event_type_2"),
            col("snap_CA"),
            col("snap_TX"),
            col("snap_WI"),
        )
        .withColumn(
            "day_id",
            expr("cast(regexp_replace(day_id_str, 'd_', '') as int)"),
        )
        .drop("day_id_str")
    )

    # --- 3. Join melted sales with calendar on day_id ---
    result = melted.join(calendar, on="day_id", how="left")

    # --- 4. Join with sell_prices on (store_id, item_id, wm_yr_wk) ---
    result = result.join(
        sell_prices_raw,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    # Select final columns in a clean order
    final_cols = [
        "day_id",
        "date",
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "sales",
        "sell_price",
        "wm_yr_wk",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    result = result.select(*final_cols)

    return result
