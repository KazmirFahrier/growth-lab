"""Prediction calibration monitoring.

Tracks whether predicted churn probabilities remain well-calibrated over time
by comparing binned prediction means against observed outcome rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd


@dataclass
class CalibrationReport:
    """Binned calibration check result."""

    checked_at: datetime
    n_samples: int
    ece: float  # Expected Calibration Error
    bins: list[dict[str, object]] = field(default_factory=list)
    is_calibrated: bool = True  # False if ECE > threshold
    warning: str = ""


class CalibrationMonitor:
    """Monitor prediction calibration over time."""

    def __init__(self, n_bins: int = 10, ece_threshold: float = 0.05):
        self._n_bins = n_bins
        self._ece_threshold = ece_threshold

    def check(
        self, y_prob: np.ndarray, y_true: np.ndarray
    ) -> CalibrationReport:
        """Compute binned calibration error.

        Args:
            y_prob: Predicted churn probabilities [0, 1].
            y_true: Observed binary outcomes.
        """
        now = datetime.now(timezone.utc)
        n = len(y_prob)

        if n < self._n_bins * 2:
            return CalibrationReport(
                checked_at=now,
                n_samples=n,
                ece=0.0,
                warning="Too few samples for calibration check.",
            )

        df = pd.DataFrame({"prob": y_prob, "label": y_true})
        df["bin"] = pd.cut(df["prob"], bins=self._n_bins, include_lowest=True)
        summary = (
            df.groupby("bin", observed=False)
            .agg(mean_pred=("prob", "mean"), actual_rate=("label", "mean"), count=("label", "size"))
            .dropna()
        )

        if len(summary) == 0:
            return CalibrationReport(checked_at=now, n_samples=n, ece=0.0)

        ece = float(
            (summary["mean_pred"] - summary["actual_rate"]).abs().mean()
        )
        bins_data = summary.reset_index().to_dict(orient="records")

        return CalibrationReport(
            checked_at=now,
            n_samples=n,
            ece=ece,
            bins=bins_data,
            is_calibrated=ece <= self._ece_threshold,
            warning=(
                "" if ece <= self._ece_threshold
                else f"ECE {ece:.4f} exceeds threshold {self._ece_threshold}"  # noqa: E501
            ),
        )


def calibration_report(
    monitor: CalibrationMonitor, y_prob: np.ndarray, y_true: np.ndarray
) -> CalibrationReport:
    """Convenience: run a calibration check."""
    return monitor.check(y_prob, y_true)
