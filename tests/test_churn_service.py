"""Security and readiness tests for the churn service boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from growth_lab.churn.features import ALL_FEATURE_NAMES
from growth_lab.serve.app import create_app
from growth_lab.service import Settings

API_KEY = "c" * 32
REQUEST = {
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


@pytest.fixture()
def artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_path = tmp_path / "churn_model.joblib"
    names_path = tmp_path / "feature_names.json"
    card_path = tmp_path / "model_card.json"
    features = pd.DataFrame(
        np.vstack([np.zeros(16), np.ones(16), np.full(16, 0.25), np.full(16, 0.75)]),
        columns=ALL_FEATURE_NAMES,
    )
    labels = np.array([0, 1, 0, 1])
    model = LogisticRegression(random_state=42).fit(features, labels)
    joblib.dump(model, model_path)
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    names_path.write_text(json.dumps(ALL_FEATURE_NAMES))
    card_path.write_text(
        json.dumps(
            {
                "model_version": "1.0.0",
                "training_date": "2026-08-01T00:00:00+00:00",
                "n_users": 4,
                "churn_rate": 0.5,
                "best_model": "logistic",
                "best_test_auc": 1.0,
                "final_test_auc": 1.0,
                "selection_method": "temporal_cross_validation",
                "selection_metric": "roc_auc",
                "selection_folds": 3,
                "selection_cv_mean_auc": 0.9,
                "selection_cv_std_auc": 0.04082482904638629,
                "candidate_cv_auc": {
                    "logistic": {
                        "mean": 0.9,
                        "std": 0.04082482904638629,
                        "fold_scores": [0.85, 0.9, 0.95],
                    }
                },
                "test_metrics": {"logistic_roc_auc": 1.0},
                "model_sha256": model_sha256,
                "feature_names": ALL_FEATURE_NAMES,
            }
        )
    )
    return model_path, names_path, card_path


@pytest.fixture()
def client(artifacts: tuple[Path, Path, Path]) -> Iterator[TestClient]:
    app = create_app(
        Settings(environment="test", api_key=API_KEY),
        model_path=artifacts[0],
        feature_names_path=artifacts[1],
        model_card_path=artifacts[2],
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_readiness_are_public(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"ready": True}


def test_prediction_requires_authentication(client: TestClient) -> None:
    assert client.post("/predict", json=REQUEST).status_code == 401
    response = client.post("/predict", headers={"X-API-Key": API_KEY}, json=REQUEST)
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0


def test_schema_rejects_unknown_categories_and_fields(client: TestClient) -> None:
    invalid = {**REQUEST, "channel": "affiliate"}
    assert client.post("/predict", headers={"X-API-Key": API_KEY}, json=invalid).status_code == 422
    extra = {**REQUEST, "raw_score": 1.0}
    assert client.post("/predict", headers={"X-API-Key": API_KEY}, json=extra).status_code == 422


def test_model_metadata_and_metrics_are_protected(client: TestClient) -> None:
    assert client.get("/model").status_code == 401
    metadata = client.get("/model", headers={"X-API-Key": API_KEY})
    assert metadata.status_code == 200
    assert metadata.json()["feature_names"] == ALL_FEATURE_NAMES
    assert metadata.json()["selection_method"] == "temporal_cross_validation"
    assert metadata.json()["selection_cv_mean_auc"] == 0.9
    assert metadata.json()["final_test_auc"] == 1.0
    metrics = client.get("/metrics", headers={"X-API-Key": API_KEY})
    assert metrics.status_code == 200
    assert "churn_predictions_total" in metrics.text


def test_missing_bundle_fails_readiness(tmp_path: Path) -> None:
    app = create_app(
        Settings(environment="test"),
        model_path=tmp_path / "missing.joblib",
        feature_names_path=tmp_path / "missing.json",
        model_card_path=tmp_path / "missing-card.json",
    )
    with TestClient(app) as local:
        assert local.get("/health").status_code == 200
        assert local.get("/ready").status_code == 503


def test_tampered_model_fails_before_deserialization(
    artifacts: tuple[Path, Path, Path],
) -> None:
    artifacts[0].write_bytes(artifacts[0].read_bytes() + b"tampered")
    app = create_app(
        Settings(environment="test"),
        model_path=artifacts[0],
        feature_names_path=artifacts[1],
        model_card_path=artifacts[2],
    )
    with TestClient(app) as local:
        assert local.get("/ready").status_code == 503


def test_model_card_requires_selection_evidence(artifacts: tuple[Path, Path, Path]) -> None:
    card = json.loads(artifacts[2].read_text())
    del card["candidate_cv_auc"]
    artifacts[2].write_text(json.dumps(card))
    app = create_app(
        Settings(environment="test"),
        model_path=artifacts[0],
        feature_names_path=artifacts[1],
        model_card_path=artifacts[2],
    )
    with TestClient(app) as local:
        assert local.get("/ready").status_code == 503
