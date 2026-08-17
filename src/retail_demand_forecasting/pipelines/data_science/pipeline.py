"""Data Science pipeline — model training and evaluation."""

from kedro.pipeline import Pipeline, node

from ..nodes.data_science import train_model


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data science pipeline.

    Trains a RandomForest regressor on the engineered features,
    evaluates on a held-out test split, and logs metrics/artifacts
    to MLflow.

    Returns:
        Kedro Pipeline with a single ``train_model_node``.
    """
    return Pipeline(
        [
            node(
                func=train_model,
                inputs=["model_input_features", "params:feature_engineering"],
                outputs="model_metrics",
                name="train_model_node",
            ),
        ]
    )
