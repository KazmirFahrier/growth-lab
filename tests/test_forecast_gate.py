"""Phase 4 gate (forecasting): models must beat seasonal-naive out-of-sample.

MASE >= 1 fails CI — a model that cannot beat "same day last week" has not
earned its complexity.
"""

from __future__ import annotations

import numpy as np
import pytest

from growth_lab.forecasting import (
    BoostedStumpsForecaster,
    HoltWinters,
    SeasonalNaive,
    bottom_up_forecast,
    rolling_origin_backtest,
)
from growth_lab.simulator.scenarios import daily_series

FIRST_ORIGIN = 400
HORIZON = 14


@pytest.fixture(scope="module")
def scenario():  # type: ignore[no-untyped-def]
    return daily_series()


def test_holt_winters_beats_baseline(scenario) -> None:  # type: ignore[no-untyped-def]
    result = rolling_origin_backtest(HoltWinters(), scenario.total, FIRST_ORIGIN, HORIZON)
    assert result.n_folds >= 5
    assert result.beats_baseline(), f"Holt-Winters MASE {result.mase:.3f} >= 1"
    assert result.mase < 0.8, "HW should beat naive comfortably on a trending series"


def test_boosted_stumps_beat_baseline(scenario) -> None:  # type: ignore[no-untyped-def]
    result = rolling_origin_backtest(
        BoostedStumpsForecaster(), scenario.total, FIRST_ORIGIN, HORIZON
    )
    assert result.beats_baseline(), f"boosted stumps MASE {result.mase:.3f} >= 1"


def test_quantile_forecasts_are_ordered_and_useful(scenario) -> None:  # type: ignore[no-untyped-def]
    result = rolling_origin_backtest(HoltWinters(), scenario.total, FIRST_ORIGIN, HORIZON)
    naive = rolling_origin_backtest(SeasonalNaive(), scenario.total, FIRST_ORIGIN, HORIZON)
    # median pinball must beat the baseline's median pinball
    assert result.pinball[0.5] < naive.pinball[0.5]
    # tail losses are necessarily smaller than median losses in pinball units
    assert result.pinball[0.1] < result.pinball[0.5]
    assert result.pinball[0.9] < result.pinball[0.5]


def test_seasonal_naive_mase_is_one_by_construction(scenario) -> None:  # type: ignore[no-untyped-def]
    result = rolling_origin_backtest(SeasonalNaive(), scenario.total, FIRST_ORIGIN, HORIZON)
    assert result.mase == pytest.approx(1.0)


def test_backtest_never_leaks_future_data(scenario) -> None:  # type: ignore[no-untyped-def]
    """A spy model records every history it is fitted on. The harness also
    fits shorter quantile probes, so the checks are: every expected training
    window occurs, and no fit ever sees beyond the last fold's origin."""
    lengths: list[int] = []

    class Spy(SeasonalNaive):
        def fit(self, history) -> None:  # type: ignore[no-untyped-def]
            lengths.append(len(history))
            super().fit(history)

    rolling_origin_backtest(Spy(), scenario.total, FIRST_ORIGIN, HORIZON, step=28)
    expected = set(range(FIRST_ORIGIN, len(scenario.total) - HORIZON + 1, 28))
    assert expected.issubset(set(lengths))
    assert max(lengths) <= max(expected)


def test_bottom_up_hierarchy_is_coherent_and_accurate(scenario) -> None:  # type: ignore[no-untyped-def]
    train = scenario.series[:FIRST_ORIGIN]
    actual_total = scenario.total[FIRST_ORIGIN : FIRST_ORIGIN + HORIZON]

    forecast = bottom_up_forecast(HoltWinters, train, scenario.channels, HORIZON)
    assert np.allclose(forecast.per_channel.sum(axis=1), forecast.total)

    naive = SeasonalNaive()
    naive.fit(scenario.total[:FIRST_ORIGIN])
    naive_mae = float(np.mean(np.abs(actual_total - naive.predict(HORIZON))))
    bu_mae = float(np.mean(np.abs(actual_total - forecast.total)))
    assert bu_mae < naive_mae, "bottom-up total should beat naive on the total"
