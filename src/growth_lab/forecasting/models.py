"""Forecasters: seasonal-naive baseline, Holt-Winters, boosted stumps.

All models implement the same protocol — fit(history), predict(horizon) —
and all of them assume day 0 of the history is a Monday (positions map to
day-of-week as t % 7). The baseline is not a strawman: every other model
must beat it out-of-sample or fail CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

SEASON = 7


class Forecaster(Protocol):
    def fit(self, history: FloatArray) -> None: ...

    def predict(self, horizon: int) -> FloatArray: ...


@dataclass
class SeasonalNaive:
    """Forecast = the most recent observation for the same day of week."""

    _history: FloatArray = field(default_factory=lambda: np.empty(0))

    def fit(self, history: FloatArray) -> None:
        if len(history) < SEASON:
            raise ValueError(f"need at least {SEASON} observations")
        self._history = history

    def predict(self, horizon: int) -> FloatArray:
        last_season = self._history[-SEASON:]
        reps = int(np.ceil(horizon / SEASON))
        result: FloatArray = np.tile(last_season, reps)[:horizon]
        return result


@dataclass
class HoltWinters:
    """Additive Holt-Winters with weekly seasonality; smoothing parameters
    chosen by grid search on in-sample one-step error."""

    alpha_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
    beta_grid: tuple[float, ...] = (0.01, 0.05, 0.1)
    gamma_grid: tuple[float, ...] = (0.05, 0.1, 0.3)
    _level: float = 0.0
    _trend: float = 0.0
    _seasonal: FloatArray = field(default_factory=lambda: np.zeros(SEASON))
    _steps_seen: int = 0

    def _run(
        self, history: FloatArray, alpha: float, beta: float, gamma: float
    ) -> tuple[float, float, FloatArray, float]:
        n = len(history)
        level = float(history[:SEASON].mean())
        trend = float((history[SEASON : 2 * SEASON].mean() - history[:SEASON].mean()) / SEASON)
        seasonal = (history[:SEASON] - level).astype(np.float64).copy()
        sse = 0.0
        for t in range(n):
            s_idx = t % SEASON
            forecast = level + trend + seasonal[s_idx]
            error = float(history[t]) - forecast
            if t >= 2 * SEASON:  # skip the initialization window
                sse += error * error
            prev_level = level
            level = alpha * (float(history[t]) - seasonal[s_idx]) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasonal[s_idx] = gamma * (float(history[t]) - level) + (1 - gamma) * seasonal[s_idx]
        return level, trend, seasonal, sse

    def fit(self, history: FloatArray) -> None:
        if len(history) < 3 * SEASON:
            raise ValueError(f"need at least {3 * SEASON} observations")
        best_sse = np.inf
        for alpha in self.alpha_grid:
            for beta in self.beta_grid:
                for gamma in self.gamma_grid:
                    level, trend, seasonal, sse = self._run(history, alpha, beta, gamma)
                    if sse < best_sse:
                        best_sse = sse
                        self._level, self._trend, self._seasonal = level, trend, seasonal
                        self._steps_seen = len(history)
        if not np.isfinite(best_sse):
            raise RuntimeError("Holt-Winters failed to fit any parameter combination")

    def predict(self, horizon: int) -> FloatArray:
        steps = np.arange(1, horizon + 1)
        seasonal_idx = (self._steps_seen + steps - 1) % SEASON
        result: FloatArray = self._level + self._trend * steps + self._seasonal[seasonal_idx]
        return result


@dataclass
class _Stump:
    feature: int
    threshold: float
    left_value: float
    right_value: float


@dataclass
class BoostedStumpsForecaster:
    """Gradient boosting on depth-1 trees over lag/calendar features,
    forecasting recursively. Implemented from scratch: squared loss, stump
    chosen per round by exhaustive quantile-threshold search."""

    n_rounds: int = 300
    learning_rate: float = 0.1
    lags: tuple[int, ...] = (1, 7, 14)
    _stumps: list[_Stump] = field(default_factory=list)
    _init: float = 0.0
    _history: FloatArray = field(default_factory=lambda: np.empty(0))

    def _features(self, values: FloatArray, t: int) -> FloatArray:
        row = [float(values[t - lag]) for lag in self.lags]
        dow_onehot = [1.0 if t % SEASON == d else 0.0 for d in range(1, SEASON)]
        return np.array(row + dow_onehot + [float(t)])

    def fit(self, history: FloatArray) -> None:
        max_lag = max(self.lags)
        n = len(history)
        if n < max_lag + 2 * SEASON:
            raise ValueError("history too short for the configured lags")
        x = np.vstack([self._features(history, t) for t in range(max_lag, n)])
        y = history[max_lag:]

        self._stumps = []
        self._init = float(y.mean())
        pred: FloatArray = np.full(len(y), self._init, dtype=np.float64)
        for _ in range(self.n_rounds):
            residual = y - pred
            best_sse = float(np.inf)
            best: _Stump | None = None
            for j in range(x.shape[1]):
                column = x[:, j]
                for q in np.linspace(0.1, 0.9, 9):
                    threshold = float(np.quantile(column, q))
                    left = column <= threshold
                    if left.all() or not left.any():
                        continue
                    lv = float(residual[left].mean())
                    rv = float(residual[~left].mean())
                    sse = float(((residual - np.where(left, lv, rv)) ** 2).sum())
                    if sse < best_sse:
                        best_sse = sse
                        best = _Stump(j, threshold, lv, rv)
            if best is None:
                break
            contribution: FloatArray = np.where(
                x[:, best.feature] <= best.threshold, best.left_value, best.right_value
            ).astype(np.float64)
            pred = (pred + self.learning_rate * contribution).astype(np.float64)
            self._stumps.append(best)
        self._history = history.astype(np.float64).copy()

    def _predict_row(self, row: FloatArray) -> float:
        value = self._init
        for s in self._stumps:
            value += self.learning_rate * (
                s.left_value if row[s.feature] <= s.threshold else s.right_value
            )
        return value

    def predict(self, horizon: int) -> FloatArray:
        n = len(self._history)
        values: FloatArray = np.concatenate([self._history, np.zeros(horizon)])
        for j in range(horizon):
            t = n + j
            values[t] = self._predict_row(self._features(values, t))
        result: FloatArray = values[n:]
        return result
