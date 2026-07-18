"""Risk & monitoring: anomaly detection, probability calibration, drift.

This package is sealed: it never imports the simulator or reads ground
truth. Detection quality and calibration are scored externally in
tests/test_risk_gate.py.
"""

from growth_lab.risk.anomaly import (
    IsolationForest,
    ResidualAnomalies,
    decompose,
    isolation_forest_detector,
    mad_residual_detector,
)
from growth_lab.risk.calibration import (
    ReliabilityBin,
    ThresholdChoice,
    auc,
    brier_score,
    cost_optimal_threshold,
    expected_calibration_error,
    reliability_curve,
)
from growth_lab.risk.drift import PSI_ALARM, PSI_WATCH, DriftReport, population_stability_index

__all__ = [
    "PSI_ALARM",
    "PSI_WATCH",
    "DriftReport",
    "IsolationForest",
    "ReliabilityBin",
    "ResidualAnomalies",
    "ThresholdChoice",
    "auc",
    "brier_score",
    "cost_optimal_threshold",
    "decompose",
    "expected_calibration_error",
    "isolation_forest_detector",
    "mad_residual_detector",
    "population_stability_index",
    "reliability_curve",
]
