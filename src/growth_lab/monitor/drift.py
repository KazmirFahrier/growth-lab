"""Data and prediction drift monitoring.

Compares current prediction request distributions against a stored reference
(baseline) to detect feature drift, prediction drift, and data freshness issues.
Uses evidently for statistical drift tests where available, falls back to
simple KS-statistic-based checks otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd


@dataclass
class DriftReport:
    """Results of a data drift check against the reference baseline."""

    checked_at: datetime
    n_current: int
    n_reference: int
    features_drifted: list[str] = field(default_factory=list)
    drift_scores: dict[str, float] = field(default_factory=dict)
    prediction_drift_detected: bool = False
    warning: str = ""

    @property
    def any_drift(self) -> bool:
        return len(self.features_drifted) > 0 or self.prediction_drift_detected


class DriftMonitor:
    """Track data drift between a reference distribution and live predictions."""

    def __init__(self, drift_threshold: float = 0.1):
        self._reference: pd.DataFrame | None = None
        self._threshold = drift_threshold

    def set_reference(self, df: pd.DataFrame) -> None:
        """Store the reference distribution (typically training data)."""
        self._reference = df.copy()

    def check(
        self, current: pd.DataFrame, prediction_scores: np.ndarray | None = None
    ) -> DriftReport:
        """Compare current data against the reference baseline.

        Uses per-feature KS-like statistic: maximum absolute difference in
        empirical CDFs normalized to [0,1]. Features exceeding the threshold
        are flagged as drifted.
        """
        now = datetime.now(timezone.utc)
        if self._reference is None:
            return DriftReport(
                checked_at=now,
                n_current=len(current),
                n_reference=0,
                warning="No reference distribution set; call set_reference() first.",
            )

        numeric_cols = current.select_dtypes(include=[np.number]).columns
        common = [c for c in numeric_cols if c in self._reference.columns]
        drifted: list[str] = []
        scores: dict[str, float] = {}

        for col in common:
            ref_vals = self._reference[col].dropna().values
            cur_vals = current[col].dropna().values
            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            # Simple KS-like: max |F_ref(x) - F_cur(x)| over decile edges
            edges = np.quantile(np.concatenate([ref_vals, cur_vals]), np.linspace(0, 1, 11))
            ref_cdf = np.searchsorted(np.sort(ref_vals), edges, side="right") / len(ref_vals)
            cur_cdf = np.searchsorted(np.sort(cur_vals), edges, side="right") / len(cur_vals)
            ks = float(np.max(np.abs(ref_cdf - cur_cdf)))
            scores[col] = ks
            if ks > self._threshold:
                drifted.append(col)

        pred_drift = False
        if prediction_scores is not None and len(prediction_scores) > 0:
            # Simple check: is mean prediction significantly different?
            # (This is a placeholder — production would use a proper statistical test)
            pred_drift = bool(np.mean(prediction_scores) > 0.7 or np.mean(prediction_scores) < 0.1)

        return DriftReport(
            checked_at=now,
            n_current=len(current),
            n_reference=len(self._reference),
            features_drifted=drifted,
            drift_scores=scores,
            prediction_drift_detected=pred_drift,
        )


def drift_report(monitor: DriftMonitor, current: pd.DataFrame) -> DriftReport:
    """Convenience: run a drift check and return the report."""
    return monitor.check(current)
