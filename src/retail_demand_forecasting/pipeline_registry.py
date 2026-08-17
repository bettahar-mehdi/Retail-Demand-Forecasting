"""Pipeline registration."""
from typing import Dict
from kedro.pipeline import Pipeline

from .pipelines.data_engineering import create_pipeline as data_engineering_pipeline
from .pipelines.feature_engineering import create_pipeline as feature_engineering_pipeline
from .pipelines.data_science import create_pipeline as data_science_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""
    return {
        "__default__": (
            data_engineering_pipeline()
            + feature_engineering_pipeline()
            + data_science_pipeline()
        ),
        "data_engineering": data_engineering_pipeline(),
        "feature_engineering": feature_engineering_pipeline(),
        "data_science": data_science_pipeline(),
    }
