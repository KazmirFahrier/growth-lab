"""FastAPI churn prediction service.

Serves the trained churn model with validated schemas, Prometheus metrics,
and health checks. Loads the model artifact from models/churn_model.joblib.

Usage:
  growth-lab-serve
  uvicorn growth_lab.serve.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
import warnings
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

from growth_lab.serve.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
)

# ── logging ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "churn_model.joblib"
MODEL_VERSION = "0.1.0"
FEATURE_NAMES = [
    "tenure_days",
    "txn_count_obs",
    "total_spend_obs",
    "avg_txn_amount_obs",
    "days_since_last_txn",
    "txn_freq_monthly",
    "had_fraud_obs",
    "signup_dow",
    "signup_month",
    "channel_display",
    "channel_organic",
    "channel_search",
    "channel_social",
    "channel_video",
    "plan_pro",
]

# ── Prometheus metrics ───────────────────────────────────────────────────
PREDICTION_COUNT = Counter(
    "churn_predictions_total", "Total predictions served", ["outcome"]
)
PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds", "Prediction latency in seconds"
)
MODEL_LOADED = Gauge("churn_model_loaded", "Whether a model is loaded (1=yes, 0=no)")
PREDICTION_ERRORS = Counter(
    "churn_prediction_errors_total", "Total prediction errors"
)
REQUEST_SIZE = Histogram(
    "churn_request_size", "Batch request size"
)

# ── model loading ────────────────────────────────────────────────────────
_model: Optional[object] = None
_model_loaded_at: Optional[datetime] = None
_start_time = time.time()


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> None:
    """Load the churn model artifact. Call once at startup."""
    global _model, _model_loaded_at
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}; service starts in no-model mode")
        MODEL_LOADED.set(0)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _model = joblib.load(model_path)
    _model_loaded_at = datetime.now(timezone.utc)
    MODEL_LOADED.set(1)
    logger.info(f"Model loaded from {model_path} (v{MODEL_VERSION})")


# ── FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(
    title="growth-lab Churn Prediction API",
    description="Production churn risk scoring for the Meridian subscription business.",
    version=MODEL_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    load_model()


# ── Prometheus metrics endpoint ──────────────────────────────────────────
@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type="text/plain")


# ── health check ─────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy" if _model is not None else "no_model_loaded",
        model_version=MODEL_VERSION,
        model_loaded_at=_model_loaded_at or datetime.min.replace(tzinfo=timezone.utc),
        uptime_seconds=time.time() - _start_time,
    )


@app.get("/ready")
async def readiness() -> JSONResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return JSONResponse(content={"status": "ready"})


# ── model metadata ───────────────────────────────────────────────────────
@app.get("/model", response_model=ModelMetadataResponse)
async def model_metadata() -> ModelMetadataResponse:
    return ModelMetadataResponse(
        model_version=MODEL_VERSION,
        feature_names=FEATURE_NAMES,
        training_date="2025-07-01",
        n_users=0,
        churn_rate=0.0,
    )


# ── helpers ──────────────────────────────────────────────────────────────
def _build_feature_vector(req: PredictionRequest) -> np.ndarray:
    """Convert a PredictionRequest into the feature vector expected by the model."""
    # Numeric features
    feats = [
        req.tenure_days,
        req.txn_count_obs,
        req.total_spend_obs,
        req.avg_txn_amount_obs,
        req.days_since_last_txn,
        req.txn_freq_monthly,
        req.had_fraud_obs,
        req.signup_dow,
        req.signup_month,
    ]
    # Categorical dummies (order must match training)
    channel_dummies = {
        "display": [1, 0, 0, 0, 0],
        "organic": [0, 1, 0, 0, 0],
        "search":  [0, 0, 1, 0, 0],
        "social":  [0, 0, 0, 1, 0],
        "video":   [0, 0, 0, 0, 1],
    }
    plan_dummies = {"basic": [0], "pro": [1]}

    ch = req.channel.lower()
    feats.extend(channel_dummies.get(ch, [0, 0, 0, 0, 0]))
    feats.extend(plan_dummies.get(req.plan.lower(), [0]))

    return np.array(feats, dtype=np.float64).reshape(1, -1)


def _predict_one(req: PredictionRequest) -> PredictionResponse:
    """Score a single user."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with PREDICTION_LATENCY.time():
        try:
            X = _build_feature_vector(req)
            prob = float(_model.predict_proba(X)[0, 1])
            pred = bool(prob >= 0.5)
        except Exception:
            PREDICTION_ERRORS.inc()
            logger.exception(f"Prediction failed for user {req.user_id}")
            raise

    PREDICTION_COUNT.labels(outcome="churn" if pred else "retain").inc()
    return PredictionResponse(
        user_id=req.user_id,
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc),
    )


# ── prediction endpoints ─────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest) -> PredictionResponse:
    return _predict_one(req)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(req: BatchPredictionRequest) -> BatchPredictionResponse:
    REQUEST_SIZE.observe(len(req.users))
    predictions = [_predict_one(user) for user in req.users]
    return BatchPredictionResponse(predictions=predictions, model_version=MODEL_VERSION)


# ── CLI entry ────────────────────────────────────────────────────────────
def main() -> None:
    import uvicorn
    uvicorn.run("growth_lab.serve.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
