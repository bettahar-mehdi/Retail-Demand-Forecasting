"""Data Engineering pipeline — raw CSV ingestion and unpivot."""

from kedro.pipeline import Pipeline, node

from ..nodes.data_engineering import unpivot_sales


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data engineering pipeline.

    Reads raw sales, calendar, and price CSVs, then unpivots the
    wide-format sales matrix into a long-format DataFrame joined
    with calendar dates and sell prices.

    Returns:
        Kedro Pipeline with a single ``unpivot_sales_node``.
    """
    return Pipeline(
        [
            node(
                func=unpivot_sales,
                inputs=[
                    "sales_train_raw",
                    "calendar_raw",
                    "sell_prices_raw",
                ],
                outputs="intermediate_sales_melted",
                name="unpivot_sales_node",
            ),
        ]
    )
