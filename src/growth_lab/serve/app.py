"""FastAPI churn prediction service.

Serves the trained churn model with validated schemas, Prometheus metrics,
and health checks. Loads the model artifact from models/churn_model.joblib
and feature names from models/feature_names.json (written by training).

Usage:
  growth-lab-serve
  uvicorn growth_lab.serve.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
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

# ── paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "churn_model.joblib"
FEATURE_NAMES_PATH = REPO_ROOT / "models" / "feature_names.json"
MODEL_VERSION = "0.2.0"

# ── categorical levels (must match features.py) ──────────────────────────
CHANNEL_LEVELS = ["display", "organic", "search", "social", "video"]
PLAN_LEVELS = ["basic", "pro"]


def _load_feature_names() -> list[str]:
    """Load the canonical feature list written during training."""
    if FEATURE_NAMES_PATH.exists():
        names = json.loads(FEATURE_NAMES_PATH.read_text())
        logger.info(f"Loaded {len(names)} feature names from {FEATURE_NAMES_PATH}")
        return names
    logger.warning(f"feature_names.json not found at {FEATURE_NAMES_PATH}; using defaults")
    # Fallback — must match ALL_FEATURE_NAMES in features.py
    numeric = [
        "tenure_days", "txn_count_obs", "total_spend_obs", "avg_txn_amount_obs",
        "days_since_last_txn", "txn_freq_monthly", "had_fraud_obs",
        "signup_dow", "signup_month",
    ]
    ch = [f"channel_{c}" for c in CHANNEL_LEVELS]
    pl = [f"plan_{p}" for p in PLAN_LEVELS]
    return numeric + ch + pl


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

# ── model state ──────────────────────────────────────────────────────────
_model: object | None = None
_model_loaded_at: datetime | None = None
_feature_names: list[str] = []
_start_time = time.time()


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> None:
    """Load the churn model artifact. Call once at startup."""
    global _model, _model_loaded_at, _feature_names
    _feature_names = _load_feature_names()
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


# ── health / readiness ───────────────────────────────────────────────────
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
        feature_names=_feature_names,
        training_date="2025-07-01",
        n_users=0,
        churn_rate=0.0,
    )


# ── feature vector builder ───────────────────────────────────────────────
def _build_feature_vector(req: PredictionRequest) -> np.ndarray:
    """Convert a PredictionRequest into the feature vector expected by the model.

    Uses the canonical feature name list (loaded from feature_names.json
    or the fallback).  Column order is the list order — which must match
    the order produced by build_training_set() in features.py.
    """
    # Numeric features in order
    values: list[float] = [
        float(req.tenure_days),
        float(req.txn_count_obs),
        float(req.total_spend_obs),
        float(req.avg_txn_amount_obs),
        float(req.days_since_last_txn),
        float(req.txn_freq_monthly),
        float(req.had_fraud_obs),
        float(req.signup_dow),
        float(req.signup_month),
    ]

    # Channel one-hot (5 levels, same order as CHANNEL_LEVELS)
    ch = req.channel.lower()
    for level in CHANNEL_LEVELS:
        values.append(1.0 if ch == level else 0.0)

    # Plan one-hot (2 levels)
    pl = req.plan.lower()
    for level in PLAN_LEVELS:
        values.append(1.0 if pl == level else 0.0)

    return np.array(values, dtype=np.float64).reshape(1, -1)


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
