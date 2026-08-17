"""FastAPI serving endpoint for demand forecasting predictions."""

import logging
from contextlib import asynccontextmanager
from typing import List

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..nodes.constants import FEATURE_COLS, MLFLOW_TRACKING_URI

log = logging.getLogger(__name__)

MODEL = None


# ---------------------------------------------------------------------------
# Lifespan — load the latest MLflow model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    log.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
    try:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=["0", "1"],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs:
            run_id = runs[0].info.run_id
            MODEL = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            log.info("Loaded model from run %s", run_id)
        else:
            log.warning("No MLflow runs found — model not loaded.")
    except Exception:
        log.exception("Failed to load model from MLflow")
    yield


app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "Predict daily unit sales for Walmart M5 items using a "
        "RandomForest model trained on lag, rolling-window, calendar, "
        "and SNAP features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """Single-item prediction request."""

    lag_7: int
    lag_28: int
    rolling_mean_7: float
    rolling_mean_28: float
    day_of_week: int
    month: int
    year: int
    snap_CA: int
    snap_TX: int
    snap_WI: int
    has_event_1: int
    has_event_2: int
    sell_price: float


class PredictionResponse(BaseModel):
    """Single-item prediction response."""

    prediction: float
    features: dict


class BatchPredictionRequest(BaseModel):
    """Batch prediction request (list of items)."""

    requests: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[float]
    count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
def health():
    """Return service health and model availability."""
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(req: PredictionRequest):
    """Predict daily sales for a single item."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = req.model_dump()
        X = pd.DataFrame([features])[FEATURE_COLS]
        pred = MODEL.predict(X)[0]
        return PredictionResponse(prediction=float(pred), features=features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["inference"],
)
def predict_batch(req: BatchPredictionRequest):
    """Predict daily sales for a batch of items."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        rows = [r.model_dump() for r in req.requests]
        X = pd.DataFrame(rows)[FEATURE_COLS]
        preds = MODEL.predict(X).tolist()
        return BatchPredictionResponse(predictions=preds, count=len(preds))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
