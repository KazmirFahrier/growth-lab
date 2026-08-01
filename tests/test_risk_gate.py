"""Phase 4 gate (risk): detection scored against injected truth, probability
quality enforced as a deployment gate, drift alarms verified in both
directions.
"""

from __future__ import annotations

import numpy as np
import pytest

from growth_lab.causal._regression import add_intercept, fit_logistic, sigmoid
from growth_lab.risk import (
    auc,
    brier_score,
    cost_optimal_threshold,
    expected_calibration_error,
    isolation_forest_detector,
    mad_residual_detector,
    population_stability_index,
)
from growth_lab.simulator.scenarios import daily_series, fraud_transactions

# --- anomaly detection ------------------------------------------------------


@pytest.fixture(scope="module")
def series():  # type: ignore[no-untyped-def]
    return daily_series()


def test_mad_detector_finds_injected_anomalies(series) -> None:  # type: ignore[no-untyped-def]
    result = mad_residual_detector(series.total)
    flagged = set(result.flagged_days.tolist())
    truth = set(series.anomaly_days.tolist())
    recall = len(flagged & truth) / len(truth)
    precision = len(flagged & truth) / max(len(flagged), 1)
    assert recall >= 0.8, f"recall {recall:.2f}: missed injected anomalies"
    assert precision >= 0.8, f"precision {precision:.2f}: too many false alarms"


def test_isolation_forest_agrees_on_top_anomalies(series) -> None:  # type: ignore[no-untyped-def]
    truth = set(series.anomaly_days.tolist())
    flagged = set(isolation_forest_detector(series.total, n_flags=len(truth) + 6).tolist())
    recall = len(flagged & truth) / len(truth)
    assert recall >= 0.7, f"isolation forest recall {recall:.2f}"


def test_clean_series_raises_few_alarms() -> None:
    clean = daily_series(seed=8, n_anomalies=0)
    result = mad_residual_detector(clean.total)
    assert len(result.flagged_days) <= 2  # ~0 expected at 5 robust sigmas


# --- fraud model quality ----------------------------------------------------


@pytest.fixture(scope="module")
def fraud_split():  # type: ignore[no-untyped-def]
    s = fraud_transactions()
    half = len(s.is_fraud) // 2
    design = add_intercept(s.features)
    beta = fit_logistic(design[:half], s.is_fraud[:half].astype(np.float64))
    test_probs = sigmoid(design[half:] @ beta)
    return s, half, test_probs


def test_fraud_model_discriminates(fraud_split) -> None:  # type: ignore[no-untyped-def]
    s, half, probs = fraud_split
    assert auc(s.is_fraud[half:], probs) > 0.75


def test_fraud_model_is_calibrated(fraud_split) -> None:  # type: ignore[no-untyped-def]
    s, half, probs = fraud_split
    ece = expected_calibration_error(s.is_fraud[half:], probs)
    assert ece < 0.02, f"ECE {ece:.4f}: probabilities cannot be trusted for thresholds"
    # and strictly better than the lazy constant-base-rate predictor
    base = np.full(len(probs), s.is_fraud[:half].mean())
    assert brier_score(s.is_fraud[half:], probs) < brier_score(s.is_fraud[half:], base)


def test_cost_threshold_matches_theory(fraud_split) -> None:  # type: ignore[no-untyped-def]
    """For calibrated probabilities the optimal flagging threshold is
    c_fp / (c_fp + c_fn); the empirical minimizer must land there."""
    s, half, probs = fraud_split
    choice = cost_optimal_threshold(
        s.is_fraud[half:], probs, false_positive_cost=1.0, false_negative_cost=10.0
    )
    assert abs(choice.threshold - 1.0 / 11.0) < 0.05


def test_probabilities_track_truth(fraud_split) -> None:  # type: ignore[no-untyped-def]
    s, half, probs = fraud_split
    corr = float(np.corrcoef(probs, s.true_probability[half:])[0, 1])
    assert corr > 0.95, "fitted probabilities should track the DGP closely"


# --- drift ------------------------------------------------------------------


def test_psi_stable_on_same_population() -> None:
    ref = fraud_transactions(seed=7)
    cur = fraud_transactions(seed=9)
    velocity = ref.feature_names.index("velocity")
    report = population_stability_index(ref.features[:, velocity], cur.features[:, velocity])
    assert report.status == "stable", f"false drift alarm: PSI {report.psi:.3f}"


def test_psi_alarms_on_regime_shift() -> None:
    ref = fraud_transactions(seed=7)
    shifted = fraud_transactions(seed=9, shifted=True)
    velocity = ref.feature_names.index("velocity")
    report = population_stability_index(ref.features[:, velocity], shifted.features[:, velocity])
    assert report.status == "alarm", f"missed drift: PSI {report.psi:.3f}"
    assert report.psi > 0.25
