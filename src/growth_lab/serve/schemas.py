"""Validated churn prediction service schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """One bounded churn prediction request."""

    model_config = ConfigDict(extra="forbid")

    user_id: int = Field(ge=0)
    channel: Literal["display", "organic", "search", "social", "video"]
    plan: Literal["basic", "pro"]
    tenure_days: int = Field(ge=0, le=36_500)
    txn_count_obs: int = Field(ge=0, le=1_000_000)
    total_spend_obs: float = Field(ge=0.0, le=1_000_000_000, allow_inf_nan=False)
    avg_txn_amount_obs: float = Field(ge=0.0, le=1_000_000_000, allow_inf_nan=False)
    days_since_last_txn: int = Field(ge=0, le=36_500)
    txn_freq_monthly: float = Field(ge=0.0, le=1_000_000, allow_inf_nan=False)
    had_fraud_obs: Literal[0, 1]
    signup_dow: int = Field(ge=0, le=6)
    signup_month: int = Field(ge=1, le=12)


class BatchPredictionRequest(BaseModel):
    """A bounded prediction batch."""

    model_config = ConfigDict(extra="forbid")
    users: list[PredictionRequest] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    user_id: int
    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_prediction: bool
    model_version: str
    timestamp: datetime


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded_at: datetime | None
    uptime_seconds: float


class ModelMetadataResponse(BaseModel):
    model_version: str
    feature_names: list[str]
    training_date: str
    n_users: int
    churn_rate: float
    best_model: str
    best_test_auc: float
