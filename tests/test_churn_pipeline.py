"""Integration tests for the churn prediction pipeline.

Tests the end-to-end flow: simulate → warehouse → features → train → predict.
All tests are self-contained and deterministic (fixed seed).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from growth_lab.churn.features import build_training_set
from growth_lab.churn.train import train_pipeline
from growth_lab.serve.app import _build_feature_vector
from growth_lab.serve.schemas import PredictionRequest
from growth_lab.simulator.generate import simulate
from growth_lab.simulator.params import load_truth
from growth_lab.warehouse.load import build_all


@pytest.fixture(scope="module")
def db_path() -> Iterator[Path]:
    """Build a warehouse once for all churn tests."""
    truth = load_truth()
    sim = simulate(truth, seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Path(tmpdir) / "test_growth_lab.duckdb"
        build_all(sim, db)
        yield db


class TestFeatureEngineering:
    def test_build_training_set_returns_data(self, db_path: Path) -> None:
        ts = build_training_set(db_path, cutoff_days=60, horizon_days=30)
        assert ts.n_users > 0, "Expected some eligible users"
        assert len(ts.X) == ts.n_users
        assert len(ts.y) == ts.n_users
        assert 0.0 < ts.churn_rate < 1.0, "Churn rate should be in (0,1)"

    def test_feature_names_match_expected(self, db_path: Path) -> None:
        ts = build_training_set(db_path)
        expected = {
            "tenure_days",
            "txn_count_obs",
            "total_spend_obs",
            "avg_txn_amount_obs",
            "days_since_last_txn",
            "txn_freq_monthly",
            "had_fraud_obs",
            "signup_dow",
            "signup_month",
        }
        # Categorical dummies vary by run, but core numerics must exist
        assert expected <= set(ts.feature_names), f"Missing: {expected - set(ts.feature_names)}"

    def test_no_negative_values(self, db_path: Path) -> None:
        ts = build_training_set(db_path)
        numeric = ts.X.select_dtypes(include=[np.number])
        assert (numeric.min() >= 0).all(), "All numeric features should be non-negative"

    def test_temporal_split_is_clean(self, db_path: Path) -> None:
        """Labels come from a future window — features should not leak label info."""
        ts = build_training_set(db_path, cutoff_days=30, horizon_days=30)
        # Users with zero transactions in observation window may still have label=0
        # (they could have transactions only in the prediction window)
        zero_txn = ts.X["txn_count_obs"] == 0
        if zero_txn.any():
            # At least some users with zero obs txns should NOT be labeled churned
            assert (ts.y[zero_txn] == 0).any(), (
                "Users with zero observation-window transactions should not all be labeled churned"
            )


class TestTrainingPipeline:
    def test_train_pipeline_runs(self, db_path: Path) -> None:
        _results = train_pipeline(db_path, cutoff_days=60, horizon_days=30, register_model=False)
        card = _results.get("model_card", {})
        assert isinstance(card, dict)
        assert "best_model" in card
        assert card.get("best_test_auc", 0) > 0


class TestServingSchema:
    def test_feature_vector_shape(self) -> None:
        req = PredictionRequest(
            user_id=1,
            channel="search",
            plan="pro",
            tenure_days=120,
            txn_count_obs=4,
            total_spend_obs=79.96,
            avg_txn_amount_obs=19.99,
            days_since_last_txn=15,
            txn_freq_monthly=1.0,
            had_fraud_obs=0,
            signup_dow=2,
            signup_month=3,
        )
        vec = _build_feature_vector(req)
        assert vec.shape[1] == 16, f"Expected 16 features, got {vec.shape[1]}"

    def test_prediction_request_validation(self) -> None:
        """Invalid requests should raise validation errors."""
        with pytest.raises(ValueError):
            PredictionRequest(
                user_id=1,
                channel="search",
                plan="pro",
                tenure_days=-1,
                txn_count_obs=0,
                total_spend_obs=0,
                avg_txn_amount_obs=0,
                days_since_last_txn=0,
                txn_freq_monthly=0,
                had_fraud_obs=0,
                signup_dow=0,
                signup_month=1,
            )


class TestMonitoring:
    def test_drift_monitor(self) -> None:
        from growth_lab.monitor.drift import DriftMonitor

        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(5, 2, 100)})
        cur = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(5, 2, 100)})
        monitor = DriftMonitor()
        monitor.set_reference(ref)
        report = monitor.check(cur)
        # Same distribution should show little drift
        assert not report.any_drift or all(v < 0.2 for v in report.drift_scores.values())

    def test_calibration_monitor(self) -> None:
        from growth_lab.monitor.calibration import CalibrationMonitor

        # Perfectly calibrated predictions
        y_prob = np.array([0.1] * 10 + [0.5] * 10 + [0.9] * 10)
        y_true = np.array([0] * 9 + [1] + [0] * 5 + [1] * 5 + [0] * 1 + [1] * 9)
        monitor = CalibrationMonitor()
        report = monitor.check(y_prob, y_true)
        assert report.ece >= 0.0
        assert report.n_samples == 30

        with pytest.raises(ValueError, match="equal length"):
            monitor.check(y_prob, y_true[:-1])

    def test_health_monitor(self) -> None:
        from growth_lab.monitor.health import HealthMonitor

        monitor = HealthMonitor()
        monitor.record(0.001)
        monitor.record(0.002)
        monitor.record(0.003, is_error=True)
        lat = monitor.latency_report()
        assert lat.n_requests == 3
        assert lat.error_rate > 0.0
        thr = monitor.throughput_report()
        assert thr.total_requests == 3

    def test_data_freshness_uses_warehouse_time(self, tmp_path: Path) -> None:
        from growth_lab.monitor.health import HealthMonitor

        db_path = tmp_path / "freshness.duckdb"
        con = duckdb.connect(str(db_path))
        try:
            con.execute("CREATE SCHEMA raw")
            con.execute("CREATE TABLE raw.transactions AS SELECT DATE '2026-01-02' AS txn_date")
            con.execute("CREATE TABLE raw.signups AS SELECT DATE '2026-01-01' AS signup_date")
        finally:
            con.close()

        monitor = HealthMonitor()
        fresh = monitor.data_freshness(
            db_path,
            reference_time=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )
        assert fresh.is_fresh
        assert fresh.freshness_hours == 24.0
        stale = monitor.data_freshness(
            db_path,
            reference_time=datetime(2026, 1, 5, tzinfo=timezone.utc),
        )
        assert not stale.is_fresh


class TestSealEnforcement:
    def test_churn_package_never_touches_truth(self) -> None:
        """The churn package must be covered by the existing no-truth-leak test."""
        SRC = Path(__file__).resolve().parents[1] / "src" / "growth_lab"
        FORBIDDEN = ("load_truth", "truth.yaml", "simulator.params", "users_latent", "sim_hidden")
        for path in (SRC / "churn").rglob("*.py"):
            text = path.read_text()
            for token in FORBIDDEN:
                assert token not in text, f"{path}: contains {token!r}"

    def test_serve_package_never_touches_truth(self) -> None:
        SRC = Path(__file__).resolve().parents[1] / "src" / "growth_lab"
        FORBIDDEN = ("load_truth", "truth.yaml", "simulator.params", "users_latent", "sim_hidden")
        for path in (SRC / "serve").rglob("*.py"):
            text = path.read_text()
            for token in FORBIDDEN:
                assert token not in text, f"{path}: contains {token!r}"

    def test_monitor_package_never_touches_truth(self) -> None:
        SRC = Path(__file__).resolve().parents[1] / "src" / "growth_lab"
        FORBIDDEN = ("load_truth", "truth.yaml", "simulator.params", "users_latent", "sim_hidden")
        for path in (SRC / "monitor").rglob("*.py"):
            text = path.read_text()
            for token in FORBIDDEN:
                assert token not in text, f"{path}: contains {token!r}"


class TestEndToEndServing:
    """End-to-end: train a model, load it into the serving stack, predict, verify."""

    def test_train_then_predict(self, db_path: Path, tmp_path: Path) -> None:
        import joblib

        from growth_lab.churn.train import train_pipeline
        from growth_lab.serve.app import _build_feature_vector

        # 1. Train a model and save it
        _results = train_pipeline(db_path, cutoff_days=60, horizon_days=30, register_model=False)
        model_path = tmp_path / "churn_model.joblib"
        model_src = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"
        assert model_src.exists(), "Training should produce models/churn_model.joblib"
        joblib.dump(joblib.load(model_src), model_path)

        # 2. Load the model (simulating what serve/app.py does at startup)
        model = joblib.load(model_path)

        # 3. Build a feature vector and predict
        req = PredictionRequest(
            user_id=1,
            channel="search",
            plan="pro",
            tenure_days=120,
            txn_count_obs=4,
            total_spend_obs=79.96,
            avg_txn_amount_obs=19.99,
            days_since_last_txn=15,
            txn_freq_monthly=1.0,
            had_fraud_obs=0,
            signup_dow=2,
            signup_month=3,
        )
        X = _build_feature_vector(req)
        prob = model.predict_proba(X)[0, 1]

        # 4. Verify output shape and range
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0, 1]"
        assert X.shape[1] == 16, f"Expected 16 features, got {X.shape[1]}"
