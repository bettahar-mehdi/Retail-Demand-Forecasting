"""FastAPI serving endpoint for demand forecasting predictions."""

import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..nodes.constants import FEATURE_COLS, MLFLOW_TRACKING_URI

log = logging.getLogger(__name__)

MODEL = None


# ---------------------------------------------------------------------------
# Lifespan — load model: prefer local pickle (DVC), fallback to MLflow
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    # 1. Try local DVC-tracked pickle
    local_paths = [
        Path("models/forecast_model.pkl"),
        Path(__file__).parents[3] / "models" / "forecast_model.pkl",
    ]
    for p in local_paths:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    MODEL = pickle.load(f)
                log.info("Loaded model from local pickle %s", p)
                break
            except Exception:
                log.exception("Failed to load local pickle %s", p)
    # 2. Fallback to MLflow registry
    if MODEL is None:
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
                # Try model_best first, then model
                for artifact in ["model_best", "model"]:
                    try:
                        MODEL = mlflow.sklearn.load_model(f"runs:/{run_id}/{artifact}")
                        log.info("Loaded model from MLflow run %s artifact %s", run_id, artifact)
                        break
                    except Exception:
                        continue
            else:
                log.warning("No MLflow runs found — model not loaded.")
        except Exception:
            log.exception("Failed to load model from MLflow")
    yield


app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "Predict daily unit sales for Walmart M5 items using a "
        "Tweedie/Poisson LightGBM for zero-inflated demand, with hierarchical "
        "store/dept/cat trends and recursive multi-step forecasting."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas — Phase 1 + hierarchical, float-preserving
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """Single-item prediction request with bounds validation — floats preserved (no int cast)."""

    # Lags (allow float to propagate recursive predictions like 0.15)
    lag_1: float = Field(..., ge=0, le=10000, description="Sales 1 day ago")
    lag_2: float = Field(..., ge=0, le=10000, description="Sales 2 days ago")
    lag_3: float = Field(..., ge=0, le=10000, description="Sales 3 days ago")
    lag_7: float = Field(..., ge=0, le=10000, description="Sales 7 days ago")
    lag_14: float = Field(..., ge=0, le=10000, description="Sales 14 days ago")
    lag_21: float = Field(..., ge=0, le=10000, description="Sales 21 days ago")
    lag_28: float = Field(..., ge=0, le=10000, description="Sales 28 days ago")
    # Rolling 7
    rolling_mean_7: float = Field(
        ..., ge=0, le=10000, description="7-day rolling mean shifted by 1"
    )
    rolling_min_7: float = Field(..., ge=0, le=10000)
    rolling_max_7: float = Field(..., ge=0, le=10000)
    rolling_std_7: float = Field(..., ge=0, le=10000)
    # Rolling 28
    rolling_mean_28: float = Field(
        ..., ge=0, le=10000, description="28-day rolling mean shifted by 1"
    )
    rolling_min_28: float = Field(..., ge=0, le=10000)
    rolling_max_28: float = Field(..., ge=0, le=10000)
    rolling_std_28: float = Field(..., ge=0, le=10000)
    # Hierarchical store/dept/cat trends (sparse HOBBIES_1_003 inherits store trend)
    store_rolling_mean_7: float = Field(0.0, ge=0, le=10000)
    store_rolling_mean_28: float = Field(0.0, ge=0, le=10000)
    dept_rolling_mean_7: float = Field(0.0, ge=0, le=10000)
    dept_rolling_mean_28: float = Field(0.0, ge=0, le=10000)
    cat_rolling_mean_7: float = Field(0.0, ge=0, le=10000)
    cat_rolling_mean_28: float = Field(0.0, ge=0, le=10000)
    # Calendar
    day_of_week: int = Field(..., ge=1, le=7, description="ISO day of week 1=Mon .. 7=Sun")
    day_of_month: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12, description="Month 1-12")
    year: int = Field(..., ge=2000, le=2030, description="Year")
    is_weekend: int = Field(..., ge=0, le=1)
    # Cyclical sin/cos
    day_of_week_sin: float = Field(..., ge=-1, le=1)
    day_of_week_cos: float = Field(..., ge=-1, le=1)
    day_of_month_sin: float = Field(..., ge=-1, le=1)
    day_of_month_cos: float = Field(..., ge=-1, le=1)
    month_sin: float = Field(..., ge=-1, le=1)
    month_cos: float = Field(..., ge=-1, le=1)
    # SNAP / events / price
    snap_CA: int = Field(..., ge=0, le=1, description="SNAP CA flag 0/1")
    snap_TX: int = Field(..., ge=0, le=1, description="SNAP TX flag 0/1")
    snap_WI: int = Field(..., ge=0, le=1, description="SNAP WI flag 0/1")
    has_event_1: int = Field(..., ge=0, le=1, description="Event type 1 flag")
    has_event_2: int = Field(..., ge=0, le=1, description="Event type 2 flag")
    sell_price: float = Field(..., ge=0, le=10000, description="Unit sell price")

    model_config = {"extra": "forbid"}


class PredictionResponse(BaseModel):
    """Single-item prediction response — float preserved for sparse items (0.15)."""

    prediction: float
    features: dict


class BatchPredictionRequest(BaseModel):
    """Batch prediction request (list of items)."""

    requests: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[float]
    count: int


# ---------------------------------------------------------------------------
# Helpers — align input to FEATURE_COLS, fill missing hierarchical with 0
# ---------------------------------------------------------------------------


def _prepare_X(features: dict) -> pd.DataFrame:
    """Build DataFrame aligned to FEATURE_COLS, missing cols filled 0.0."""
    # Start with provided features, fill missing FEATURE_COLS with 0.0
    row = {col: float(features.get(col, 0.0)) for col in FEATURE_COLS}
    # For categorical ints already in features, keep them but ensure float for model
    # Preserve int-like but as float for LightGBM
    return pd.DataFrame([row])[FEATURE_COLS].fillna(0.0)


def _get_model():
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — train pipeline or check MLflow/DVC artifacts",
        )
    return MODEL


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
def health():
    """Return service health and model availability."""
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(req: PredictionRequest):
    """Predict daily sales for a single item — returns float (no int rounding) for sparse items."""
    model = _get_model()
    try:
        features = req.model_dump()
        X = _prepare_X(features)
        pred = float(model.predict(X)[0])
        # Tweeded/Poisson can predict tiny positives for sparse HOBBIES_1_003; keep float, only clip negative
        pred = max(0.0, float(pred))
        return PredictionResponse(prediction=pred, features=features)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["inference"],
)
def predict_batch(req: BatchPredictionRequest):
    """Predict daily sales for a batch of items — float preserved."""
    model = _get_model()
    try:
        rows = [r.model_dump() for r in req.requests]
        # Align each row to FEATURE_COLS
        X = pd.DataFrame([{col: float(r.get(col, 0.0)) for col in FEATURE_COLS} for r in rows])[
            FEATURE_COLS
        ].fillna(0.0)
        preds = [max(0.0, float(p)) for p in model.predict(X).tolist()]
        return BatchPredictionResponse(predictions=preds, count=len(preds))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
