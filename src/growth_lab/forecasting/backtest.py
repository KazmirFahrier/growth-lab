"""Rolling-origin backtesting with leakage-proof slicing.

The harness owns the train/test boundary: models only ever receive the
training slice, and every metric is computed on observations the model never
saw. MASE uses the seasonal-naive forecast on the *same folds*, so "beats
the baseline" is an apples-to-apples claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from growth_lab.forecasting.models import Forecaster, SeasonalNaive

FloatArray = np.typing.NDArray[np.float64]

QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class BacktestResult:
    n_folds: int
    horizon: int
    mae: float
    rmse: float
    mase: float  # vs seasonal-naive on identical folds
    pinball: dict[float, float]  # per requested quantile

    def beats_baseline(self) -> bool:
        return self.mase < 1.0


def _pinball_loss(actual: FloatArray, forecast: FloatArray, q: float) -> float:
    diff = actual - forecast
    return float(np.mean(np.where(diff >= 0, q * diff, (q - 1.0) * diff)))


def _quantile_forecasts(
    model: Forecaster, train: FloatArray, point: FloatArray
) -> dict[float, FloatArray]:
    """Quantile paths: the median IS the point forecast; tails are the point
    plus empirical quantiles of the model's own holdout errors (estimated on
    a within-train split — a documented, refit-light approximation)."""
    holdout = min(len(train) // 5, 28)
    fit_part, check_part = train[:-holdout], train[-holdout:]
    probe = type(model)()  # fresh instance of the same forecaster class
    probe.fit(fit_part)
    errors = check_part - probe.predict(holdout)
    return {q: point if q == 0.5 else point + float(np.quantile(errors, q)) for q in QUANTILES}


def rolling_origin_backtest(
    model: Forecaster,
    series: FloatArray,
    first_origin: int,
    horizon: int = 14,
    step: int = 28,
) -> BacktestResult:
    """Walk the origin forward; at each fold fit on [0, origin) and score
    the next `horizon` observations."""
    n = len(series)
    if first_origin + horizon > n:
        raise ValueError("first origin leaves no room for a single fold")

    abs_errors: list[float] = []
    sq_errors: list[float] = []
    naive_abs_errors: list[float] = []
    pinball_sums: dict[float, list[float]] = {q: [] for q in QUANTILES}

    origin = first_origin
    n_folds = 0
    while origin + horizon <= n:
        train = series[:origin]
        actual = series[origin : origin + horizon]

        model.fit(train)
        point = model.predict(horizon)

        baseline = SeasonalNaive()
        baseline.fit(train)
        naive_point = baseline.predict(horizon)

        abs_errors.extend(np.abs(actual - point).tolist())
        sq_errors.extend(((actual - point) ** 2).tolist())
        naive_abs_errors.extend(np.abs(actual - naive_point).tolist())

        quantile_paths = _quantile_forecasts(model, train, point)
        for q, path in quantile_paths.items():
            pinball_sums[q].append(_pinball_loss(actual, path, q))

        origin += step
        n_folds += 1

    naive_mae = float(np.mean(naive_abs_errors))
    if naive_mae == 0:
        raise RuntimeError("seasonal-naive is perfect on this series; MASE undefined")
    mae = float(np.mean(abs_errors))
    return BacktestResult(
        n_folds=n_folds,
        horizon=horizon,
        mae=mae,
        rmse=float(np.sqrt(np.mean(sq_errors))),
        mase=mae / naive_mae,
        pinball={q: float(np.mean(v)) for q, v in pinball_sums.items()},
    )
