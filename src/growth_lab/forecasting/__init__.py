"""Forecasting: models, rolling-origin backtesting, hierarchy.

This package is sealed: it never imports the simulator or reads ground
truth. The gate in tests/test_forecast_gate.py requires every model to beat
seasonal-naive out-of-sample — models must earn their complexity.
"""

from growth_lab.forecasting.backtest import BacktestResult, rolling_origin_backtest
from growth_lab.forecasting.hierarchy import HierarchicalForecast, bottom_up_forecast
from growth_lab.forecasting.models import (
    BoostedStumpsForecaster,
    Forecaster,
    HoltWinters,
    SeasonalNaive,
)

__all__ = [
    "BacktestResult",
    "BoostedStumpsForecaster",
    "Forecaster",
    "HierarchicalForecast",
    "HoltWinters",
    "SeasonalNaive",
    "bottom_up_forecast",
    "rolling_origin_backtest",
]
