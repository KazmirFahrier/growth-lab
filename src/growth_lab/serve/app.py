"""Authenticated FastAPI service for churn risk scoring."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
import uuid
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from growth_lab import __version__
from growth_lab.churn.features import ALL_FEATURE_NAMES
from growth_lab.serve.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from growth_lab.service.app import BodyLimitMiddleware, RuntimeMiddleware, configure_logging
from growth_lab.service.config import Settings
from growth_lab.service.observability import RequestMetrics

FloatArray = npt.NDArray[np.float64]
DEFAULT_MODEL_PATH = Path("models/churn_model.joblib")
DEFAULT_FEATURE_NAMES_PATH = Path("models/feature_names.json")
DEFAULT_MODEL_CARD_PATH = Path("models/model_card.json")


class ProbabilityModel(Protocol):
    n_features_in_: int

    def predict_proba(self, values: pd.DataFrame) -> FloatArray: ...


@dataclass
class ChurnState:
    model: ProbabilityModel | None = None
    loaded_at: datetime | None = None
    feature_names: list[str] = field(default_factory=list)
    model_card: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChurnMetrics:
    predictions: Counter
    latency: Histogram
    loaded: Gauge
    errors: Counter
    request_size: Histogram


def _artifact_path(environment_name: str, default: Path) -> Path:
    return Path(os.environ.get(environment_name, str(default)))


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid artifact at {path}") from error


def _is_unit_interval_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
        and 0.0 <= float(value) <= 1.0
    )


def _validate_model_card(value: Any, feature_names: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("feature_names") != feature_names:
        raise ValueError("model card does not match the feature contract")
    if value.get("model_version") != __version__:
        raise ValueError("model card version does not match the service")
    if not isinstance(value.get("training_date"), str) or not value["training_date"]:
        raise ValueError("model card has no training date")
    n_users = value.get("n_users")
    churn_rate = value.get("churn_rate")
    best_auc = value.get("best_test_auc")
    final_test_auc = value.get("final_test_auc")
    best_model = value.get("best_model")
    selection_method = value.get("selection_method")
    selection_metric = value.get("selection_metric")
    selection_folds = value.get("selection_folds")
    selection_mean = value.get("selection_cv_mean_auc")
    selection_std = value.get("selection_cv_std_auc")
    candidate_scores = value.get("candidate_cv_auc")
    test_metrics = value.get("test_metrics")
    model_sha256 = value.get("model_sha256")
    if not isinstance(n_users, int) or n_users < 1:
        raise ValueError("model card has an invalid user count")
    if not isinstance(churn_rate, (int, float)) or not np.isfinite(churn_rate):
        raise ValueError("model card has an invalid churn rate")
    if not 0.0 <= float(churn_rate) <= 1.0:
        raise ValueError("model card has an invalid churn rate")
    if not isinstance(best_auc, (int, float)) or not np.isfinite(best_auc):
        raise ValueError("model card has an invalid test AUC")
    if not 0.0 <= float(best_auc) <= 1.0:
        raise ValueError("model card has an invalid test AUC")
    if not _is_unit_interval_number(final_test_auc) or not np.isclose(
        float(best_auc), float(cast(float, final_test_auc))
    ):
        raise ValueError("model card has inconsistent final test AUC")
    if not isinstance(best_model, str) or not best_model:
        raise ValueError("model card has no selected model")
    if selection_method != "temporal_cross_validation" or selection_metric != "roc_auc":
        raise ValueError("model card has an invalid selection method")
    if not isinstance(selection_folds, int) or selection_folds < 2:
        raise ValueError("model card has too few temporal validation folds")
    if not _is_unit_interval_number(selection_mean):
        raise ValueError("model card has an invalid selection mean")
    if not _is_unit_interval_number(selection_std):
        raise ValueError("model card has an invalid selection standard deviation")
    if not isinstance(candidate_scores, dict) or best_model not in candidate_scores:
        raise ValueError("model card has no candidate selection evidence")
    for candidate_name, summary in candidate_scores.items():
        if not isinstance(candidate_name, str) or not isinstance(summary, dict):
            raise ValueError("model card has invalid candidate selection evidence")
        mean = summary.get("mean")
        std = summary.get("std")
        fold_scores = summary.get("fold_scores")
        if not _is_unit_interval_number(mean) or not _is_unit_interval_number(std):
            raise ValueError("model card has invalid candidate selection evidence")
        if not isinstance(fold_scores, list) or len(fold_scores) != selection_folds:
            raise ValueError("model card has invalid candidate fold evidence")
        if not all(_is_unit_interval_number(score) for score in fold_scores):
            raise ValueError("model card has invalid candidate fold evidence")
        numeric_fold_scores = [float(score) for score in fold_scores]
        if not np.isclose(
            float(cast(float, mean)), float(np.mean(numeric_fold_scores))
        ) or not np.isclose(
            float(cast(float, std)), float(np.std(numeric_fold_scores))
        ):
            raise ValueError("model card candidate summary does not match its folds")
    selected_scores = candidate_scores[best_model]
    selected_mean = selected_scores["mean"]
    selected_std = selected_scores["std"]
    if not np.isclose(float(selected_mean), float(cast(float, selection_mean))) or not np.isclose(
        float(selected_std), float(cast(float, selection_std))
    ):
        raise ValueError("model card selection evidence is inconsistent")
    highest_mean = max(float(summary["mean"]) for summary in candidate_scores.values())
    if not np.isclose(float(selected_mean), highest_mean):
        raise ValueError("model card selected candidate is not the validation champion")
    selected_auc_key = f"{best_model}_roc_auc"
    if not isinstance(test_metrics, dict) or not _is_unit_interval_number(
        test_metrics.get(selected_auc_key)
    ):
        raise ValueError("model card has no final test metrics for the selected model")
    if not np.isclose(float(test_metrics[selected_auc_key]), float(cast(float, final_test_auc))):
        raise ValueError("model card final test evidence is inconsistent")
    if (
        not isinstance(model_sha256, str)
        or len(model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in model_sha256)
    ):
        raise ValueError("model card has an invalid model checksum")
    return value


def load_model(
    state: ChurnState,
    metrics: ChurnMetrics,
    model_path: Path,
    feature_names_path: Path,
    model_card_path: Path,
) -> None:
    """Load and validate the trusted model bundle."""
    if (
        not model_path.is_file()
        or not feature_names_path.is_file()
        or not model_card_path.is_file()
    ):
        metrics.loaded.set(0)
        return
    names = _load_json(feature_names_path)
    if names != ALL_FEATURE_NAMES:
        raise ValueError("churn artifact metadata does not match the serving contract")
    card = _validate_model_card(_load_json(model_card_path), names)
    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if not secrets.compare_digest(checksum, card["model_sha256"]):
        raise ValueError("churn model checksum does not match the model card")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = joblib.load(model_path)
    if not hasattr(loaded, "predict_proba") or getattr(loaded, "n_features_in_", None) != len(
        names
    ):
        raise ValueError("churn model does not match the serving feature contract")
    state.model = cast(ProbabilityModel, loaded)
    state.loaded_at = datetime.now(timezone.utc)
    state.feature_names = list(names)
    state.model_card = card
    metrics.loaded.set(1)


def _build_feature_vector(request: PredictionRequest) -> pd.DataFrame:
    values = {
        "tenure_days": float(request.tenure_days),
        "txn_count_obs": float(request.txn_count_obs),
        "total_spend_obs": request.total_spend_obs,
        "avg_txn_amount_obs": request.avg_txn_amount_obs,
        "days_since_last_txn": float(request.days_since_last_txn),
        "txn_freq_monthly": request.txn_freq_monthly,
        "had_fraud_obs": float(request.had_fraud_obs),
        "signup_dow": float(request.signup_dow),
        "signup_month": float(request.signup_month),
        **{
            f"channel_{name}": float(request.channel == name)
            for name in ("display", "organic", "search", "social", "video")
        },
        **{f"plan_{name}": float(request.plan == name) for name in ("basic", "pro")},
    }
    return pd.DataFrame(
        [[values[name] for name in ALL_FEATURE_NAMES]],
        columns=ALL_FEATURE_NAMES,
        dtype=np.float64,
    )


def _predict_one(
    state: ChurnState,
    metrics: ChurnMetrics,
    request: PredictionRequest,
) -> PredictionResponse:
    if state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    started = time.perf_counter()
    try:
        probability = float(state.model.predict_proba(_build_feature_vector(request))[0, 1])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("model returned an invalid probability")
    except HTTPException:
        raise
    except Exception as error:
        metrics.errors.inc()
        raise RuntimeError("prediction failed") from error
    finally:
        metrics.latency.observe(time.perf_counter() - started)
    prediction = probability >= 0.5
    metrics.predictions.labels(outcome="churn" if prediction else "retain").inc()
    return PredictionResponse(
        user_id=request.user_id,
        churn_probability=round(probability, 6),
        churn_prediction=prediction,
        model_version=__version__,
        timestamp=datetime.now(timezone.utc),
    )


def create_app(
    settings: Settings | None = None,
    model_path: Path | None = None,
    feature_names_path: Path | None = None,
    model_card_path: Path | None = None,
) -> FastAPI:
    runtime = settings or Settings.from_env()
    configure_logging(runtime.log_level)
    logger = logging.getLogger("growth_lab.churn_service")
    registry = CollectorRegistry()
    churn_metrics = ChurnMetrics(
        predictions=Counter(
            "churn_predictions_total", "Predictions served", ["outcome"], registry=registry
        ),
        latency=Histogram(
            "churn_prediction_latency_seconds", "Prediction latency", registry=registry
        ),
        loaded=Gauge("churn_model_loaded", "Whether the model is ready", registry=registry),
        errors=Counter("churn_prediction_errors_total", "Prediction errors", registry=registry),
        request_size=Histogram("churn_request_size", "Batch size", registry=registry),
    )
    http_metrics = RequestMetrics()
    state = ChurnState()
    selected_model = model_path or _artifact_path("GROWTH_LAB_CHURN_MODEL", DEFAULT_MODEL_PATH)
    selected_names = feature_names_path or _artifact_path(
        "GROWTH_LAB_FEATURE_NAMES", DEFAULT_FEATURE_NAMES_PATH
    )
    selected_card = model_card_path or _artifact_path(
        "GROWTH_LAB_MODEL_CARD", DEFAULT_MODEL_CARD_PATH
    )
    started_at = time.monotonic()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            await run_in_threadpool(
                load_model,
                state,
                churn_metrics,
                selected_model,
                selected_names,
                selected_card,
            )
        except (OSError, ValueError):
            logger.exception("churn model bundle failed validation")
            churn_metrics.loaded.set(0)
        yield

    app = FastAPI(
        title="Growth Lab Churn API",
        version=__version__,
        docs_url=None if runtime.environment == "production" else "/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.churn = state
    app.add_middleware(BodyLimitMiddleware, max_bytes=runtime.max_request_bytes)
    app.add_middleware(RuntimeMiddleware, settings=runtime, metrics=http_metrics)

    async def authorize(x_api_key: str | None = Header(default=None)) -> None:
        if runtime.api_key is None:
            return
        if x_api_key is None or not secrets.compare_digest(x_api_key, runtime.api_key):
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, error: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        logger.exception("unhandled churn request error", extra={"request_id": request_id})
        return JSONResponse(
            status_code=500,
            content={"error": "INTERNAL_ERROR", "request_id": request_id},
        )

    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            model_version=__version__,
            model_loaded_at=state.loaded_at,
            uptime_seconds=time.monotonic() - started_at,
        )

    @app.get("/ready", include_in_schema=False)
    async def readiness() -> JSONResponse:
        ready = state.model is not None
        return JSONResponse(status_code=200 if ready else 503, content={"ready": ready})

    @app.get("/metrics", include_in_schema=False)
    async def metrics(_: None = Depends(authorize)) -> Response:
        payload = http_metrics.render().encode() + generate_latest(registry)
        return Response(content=payload, media_type="text/plain; version=0.0.4")

    @app.get("/model", response_model=ModelMetadataResponse)
    async def model_metadata(_: None = Depends(authorize)) -> ModelMetadataResponse:
        if state.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return ModelMetadataResponse(
            model_version=__version__,
            feature_names=state.feature_names,
            training_date=str(state.model_card.get("training_date", "unknown")),
            n_users=int(state.model_card.get("n_users", 0)),
            churn_rate=float(state.model_card.get("churn_rate", 0.0)),
            best_model=str(state.model_card.get("best_model", "unknown")),
            best_test_auc=float(state.model_card.get("best_test_auc", 0.0)),
            final_test_auc=float(state.model_card.get("final_test_auc", 0.0)),
            selection_method=str(state.model_card.get("selection_method", "unknown")),
            selection_metric=str(state.model_card.get("selection_metric", "unknown")),
            selection_folds=int(state.model_card.get("selection_folds", 0)),
            selection_cv_mean_auc=float(state.model_card.get("selection_cv_mean_auc", 0.0)),
            selection_cv_std_auc=float(state.model_card.get("selection_cv_std_auc", 0.0)),
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: PredictionRequest,
        _: None = Depends(authorize),
    ) -> PredictionResponse:
        return await run_in_threadpool(_predict_one, state, churn_metrics, request)

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(
        request: BatchPredictionRequest,
        _: None = Depends(authorize),
    ) -> BatchPredictionResponse:
        churn_metrics.request_size.observe(len(request.users))
        predictions = await run_in_threadpool(
            lambda: [_predict_one(state, churn_metrics, user) for user in request.users]
        )
        return BatchPredictionResponse(predictions=predictions, model_version=__version__)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("growth_lab.serve.app:create_app", factory=True, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
