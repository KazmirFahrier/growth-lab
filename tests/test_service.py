"""Production service boundary tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from growth_lab.marketing import fit_mmm
from growth_lab.service import Settings, create_app
from growth_lab.simulator.scenarios import mmm_market

API_KEY = "a" * 32


@pytest.fixture(scope="module")
def service_db(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    path = tmp_path_factory.mktemp("service") / "growth.duckdb"
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE SCHEMA marts")
        con.execute(
            """
            CREATE TABLE marts.mart_daily_channel AS
            SELECT
                CAST(DATE '2026-01-01' + CAST(day AS INTEGER) AS DATE) AS date,
                'display' AS channel,
                100.0 AS spend,
                10000 AS impressions,
                500 AS clicks,
                50 AS signups,
                20 AS paid_signups,
                20 AS txns,
                1 AS fraud_txns,
                1000.0 + 10.0 * day AS revenue
            FROM range(35) AS days(day)
            """
        )
    finally:
        con.close()
    return path


@pytest.fixture(scope="module")
def service_mmm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    scenario = mmm_market()
    fit = fit_mmm(scenario.spend, scenario.revenue, scenario.day_of_week, scenario.channels)
    path = tmp_path_factory.mktemp("service-model") / "mmm.json"
    path.write_text(
        json.dumps(
            {
                "source": "service test fit",
                "channels": [
                    {
                        "name": name,
                        "beta": float(fit.beta[index]),
                        "decay": float(fit.decay[index]),
                        "half_sat": float(fit.half_sat[index]),
                        "current_daily_spend": float(np.mean(scenario.spend[:, index])),
                    }
                    for index, name in enumerate(scenario.channels)
                ],
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def client(service_db: Path, service_mmm: Path) -> Iterator[TestClient]:
    settings = Settings(
        environment="test",
        db_path=service_db,
        mmm_params_path=service_mmm,
        api_key=API_KEY,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_production_requires_a_strong_key() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        Settings(environment="production", api_key="short")


def test_health_is_public_and_correlated(client: TestClient) -> None:
    response = client.get("/healthz", headers={"X-Request-ID": "probe.123"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"] == "probe.123"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_ready_checks_both_artifacts(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "checks": {"warehouse": True, "mmm_params": True},
    }


def test_ready_fails_when_artifacts_are_missing(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            environment="test",
            db_path=tmp_path / "missing.duckdb",
            mmm_params_path=tmp_path / "missing.json",
        )
    )
    with TestClient(app) as local:
        response = local.get("/readyz")
    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_business_routes_require_authentication(client: TestClient) -> None:
    response = client.post("/v1/metrics", json={"metrics": ["revenue"]})
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid API key"


def test_metrics_endpoint_uses_governed_filters(client: TestClient) -> None:
    response = client.post(
        "/v1/metrics",
        headers={"X-API-Key": API_KEY},
        json={
            "metrics": ["revenue", "cac"],
            "by": ["channel"],
            "filters": {"channel": "display"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["data"]["row_count"] == 1
    assert body["data"]["parameters"] == ["display"]
    assert "display" not in body["data"]["sql"]


def test_raw_sql_and_unknown_dimensions_are_rejected(client: TestClient) -> None:
    raw = client.post(
        "/v1/metrics",
        headers={"X-API-Key": API_KEY},
        json={"metrics": ["revenue"], "where": "1=1"},
    )
    assert raw.status_code == 422

    dimension = client.post(
        "/v1/metrics",
        headers={"X-API-Key": API_KEY},
        json={"metrics": ["revenue"], "by": ["sim_hidden.users_latent"]},
    )
    assert dimension.status_code == 422
    assert dimension.json()["error_code"] == "INVALID_DIMENSION"


def test_filter_values_cannot_change_query_structure(client: TestClient) -> None:
    payload = "display' OR 1=1 --"
    response = client.post(
        "/v1/metrics",
        headers={"X-API-Key": API_KEY},
        json={
            "metrics": ["revenue"],
            "by": ["channel"],
            "filters": {"channel": payload},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["rows"] == []
    assert payload not in body["data"]["sql"]
    assert body["data"]["parameters"] == [payload]


def test_forecast_and_budget_endpoints(client: TestClient) -> None:
    forecast = client.post(
        "/v1/forecast",
        headers={"X-API-Key": API_KEY},
        json={"horizon_days": 7},
    )
    assert forecast.status_code == 200
    assert len(forecast.json()["data"]["daily"]) == 7

    budget = client.post(
        "/v1/budget",
        headers={"X-API-Key": API_KEY},
        json={"total_daily_budget": 1800},
    )
    assert budget.status_code == 200
    assert budget.json()["data"]["total_daily_budget"] == 1800


def test_operational_metrics_are_authenticated_and_bounded(client: TestClient) -> None:
    denied = client.get("/metrics")
    assert denied.status_code == 401
    response = client.get("/metrics", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert "growth_lab_http_requests_total" in response.text
    assert 'route="/healthz"' in response.text


def test_request_limit_is_enforced_before_validation(service_db: Path, service_mmm: Path) -> None:
    app = create_app(
        Settings(
            environment="test",
            db_path=service_db,
            mmm_params_path=service_mmm,
            api_key=API_KEY,
            max_request_bytes=1024,
        )
    )
    with TestClient(app) as local:
        response = local.post(
            "/v1/metrics",
            headers={"X-API-Key": API_KEY},
            json={"metrics": ["revenue"], "padding": "x" * 2000},
        )
    assert response.status_code == 413
    assert response.json()["error"] == "REQUEST_TOO_LARGE"
