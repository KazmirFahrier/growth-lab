"""Bottom-up hierarchical forecasting.

Forecast each child series, sum for the total — coherence (children add up
to the parent) holds by construction and is asserted anyway, loudly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from growth_lab.forecasting.models import Forecaster

FloatArray = np.typing.NDArray[np.float64]


@dataclass(frozen=True)
class HierarchicalForecast:
    channels: tuple[str, ...]
    per_channel: FloatArray  # (horizon, n_channels)
    total: FloatArray  # (horizon,)


def bottom_up_forecast(
    make_model: type[Forecaster],
    series: FloatArray,
    channels: tuple[str, ...],
    horizon: int,
) -> HierarchicalForecast:
    """Fit one model per child series; the total is their sum."""
    _, n_channels = series.shape
    if n_channels != len(channels):
        raise ValueError("channel names do not match series columns")

    per_channel = np.empty((horizon, n_channels))
    for c in range(n_channels):
        model = make_model()
        model.fit(series[:, c])
        per_channel[:, c] = model.predict(horizon)

    total = per_channel.sum(axis=1)
    if not np.allclose(per_channel.sum(axis=1), total):
        raise RuntimeError("hierarchy incoherent: children do not sum to total")
    return HierarchicalForecast(channels=channels, per_channel=per_channel, total=total)
