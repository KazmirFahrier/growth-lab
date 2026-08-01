"""Pydantic request/response schemas for the churn prediction API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Single user churn prediction request."""

    user_id: int = Field(..., description="User identifier")
    channel: str = Field(..., description="Acquisition channel (search, social, display, video, organic)")
    plan: str = Field(..., description="Subscription plan (basic or pro)")
    tenure_days: int = Field(..., ge=0, description="Days since signup")
    txn_count_obs: int = Field(..., ge=0, description="Transaction count in observation window")
    total_spend_obs: float = Field(..., ge=0.0, description="Total spend in observation window")
    avg_txn_amount_obs: float = Field(..., ge=0.0, description="Average transaction amount")
    days_since_last_txn: int = Field(..., ge=0, description="Days since last transaction")
    txn_freq_monthly: float = Field(..., ge=0.0, description="Transactions per month")
    had_fraud_obs: int = Field(..., ge=0, le=1, description="Whether user had a fraudulent transaction")
    signup_dow: int = Field(..., ge=0, le=6, description="Day of week of signup (0=Mon..6=Sun)")
    signup_month: int = Field(..., ge=1, le=12, description="Month of signup")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 42,
                "channel": "search",
                "plan": "pro",
                "tenure_days": 120,
                "txn_count_obs": 4,
                "total_spend_obs": 79.96,
                "avg_txn_amount_obs": 19.99,
                "days_since_last_txn": 15,
                "txn_freq_monthly": 1.0,
                "had_fraud_obs": 0,
                "signup_dow": 2,
                "signup_month": 3,
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch churn prediction request."""

    users: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Single prediction response."""

    user_id: int
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: bool
    model_version: str
    timestamp: datetime


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse]
    model_version: str


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str
    model_version: str
    model_loaded_at: datetime
    uptime_seconds: float


class ModelMetadataResponse(BaseModel):
    """Model metadata response."""

    model_version: str
    feature_names: list[str]
    training_date: str
    n_users: int
    churn_rate: float
