"""Feature Engineering pipeline — lag, rolling, and calendar features."""

from kedro.pipeline import Pipeline, node

from ..nodes.feature_engineering import create_features


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature engineering pipeline.

    Adds lag (7, 28-day), rolling-mean (7, 28-day), calendar
    components, and event flags using PySpark Window functions.

    Returns:
        Kedro Pipeline with a single ``create_features_node``.
    """
    return Pipeline(
        [
            node(
                func=create_features,
                inputs=["intermediate_sales_melted", "params:feature_engineering"],
                outputs="model_input_features",
                name="create_features_node",
            ),
        ]
    )
