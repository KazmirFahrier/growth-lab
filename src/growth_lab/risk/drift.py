"""Population stability index: the standard production drift alarm.

PSI < 0.1 is stable, 0.1-0.25 is worth watching, > 0.25 means the scoring
population no longer looks like the training population and the model's
calibration promises are void.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

PSI_WATCH = 0.1
PSI_ALARM = 0.25


@dataclass(frozen=True)
class DriftReport:
    psi: float
    status: str  # "stable" | "watch" | "alarm"


def population_stability_index(
    reference: FloatArray, current: FloatArray, n_bins: int = 10
) -> DriftReport:
    """PSI over reference-decile bins with epsilon-smoothed proportions."""
    if len(reference) < n_bins * 5 or len(current) < n_bins * 5:
        raise ValueError("need at least 5 observations per bin in both samples")
    edges = np.quantile(reference, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    eps = 1e-6
    ref_p = np.maximum(ref_counts / len(reference), eps)
    cur_p = np.maximum(cur_counts / len(current), eps)
    psi = float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    status = "stable" if psi < PSI_WATCH else ("watch" if psi < PSI_ALARM else "alarm")
    return DriftReport(psi=psi, status=status)
