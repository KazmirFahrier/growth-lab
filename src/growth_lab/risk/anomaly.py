"""Anomaly detection on daily series: robust residual rule + isolation forest.

The residual detector is the workhorse: remove trend (rolling median) and
weekly pattern (per-weekday median), flag residuals beyond k robust sigmas
(MAD). The isolation forest is implemented from scratch and consumes the
same decomposition as features — an unsupervised second opinion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

SEASON = 7


def _rolling_median(y: FloatArray, window: int) -> FloatArray:
    half = window // 2
    padded = np.pad(y, half, mode="edge")
    return np.array([float(np.median(padded[i : i + window])) for i in range(len(y))])


def decompose(y: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Robust trend / weekly / residual decomposition (medians throughout,
    so single-day shocks do not drag the components toward themselves)."""
    trend = _rolling_median(y, 2 * SEASON + 1)
    detrended = y - trend
    dow = np.arange(len(y)) % SEASON
    weekly = np.array([float(np.median(detrended[dow == d])) for d in range(SEASON)])[dow]
    residual = detrended - weekly
    return trend, weekly, residual


@dataclass(frozen=True)
class ResidualAnomalies:
    flagged_days: IntArray
    scores: FloatArray  # |residual| in robust sigmas
    threshold_sigmas: float


def mad_residual_detector(y: FloatArray, threshold_sigmas: float = 5.0) -> ResidualAnomalies:
    _, _, residual = decompose(y)
    mad = float(np.median(np.abs(residual - np.median(residual))))
    robust_sigma = 1.4826 * mad
    if robust_sigma == 0:
        raise ValueError("series has zero robust dispersion; detector is undefined")
    scores = np.abs(residual) / robust_sigma
    flagged: IntArray = np.flatnonzero(scores > threshold_sigmas).astype(np.int64)
    return ResidualAnomalies(
        flagged_days=flagged, scores=scores, threshold_sigmas=threshold_sigmas
    )


# --- isolation forest (from scratch) ----------------------------------------


@dataclass
class _Node:
    feature: int = -1
    split: float = 0.0
    left: _Node | None = None
    right: _Node | None = None
    size: int = 0  # leaf: number of samples that landed here


def _grow(x: FloatArray, depth: int, max_depth: int, rng: np.random.Generator) -> _Node:
    n = len(x)
    if depth >= max_depth or n <= 1:
        return _Node(size=n)
    feature = int(rng.integers(0, x.shape[1]))
    lo, hi = float(x[:, feature].min()), float(x[:, feature].max())
    if lo == hi:
        return _Node(size=n)
    split = float(rng.uniform(lo, hi))
    mask = x[:, feature] < split
    return _Node(
        feature=feature,
        split=split,
        left=_grow(x[mask], depth + 1, max_depth, rng),
        right=_grow(x[~mask], depth + 1, max_depth, rng),
    )


def _path_length(node: _Node, row: FloatArray, depth: int) -> float:
    if node.left is None or node.right is None:
        return depth + _average_bst_depth(node.size)
    child = node.left if row[node.feature] < node.split else node.right
    return _path_length(child, row, depth + 1)


def _average_bst_depth(n: int) -> float:
    if n <= 1:
        return 0.0
    harmonic = math.log(n - 1) + 0.5772156649
    return 2.0 * harmonic - 2.0 * (n - 1) / n


@dataclass
class IsolationForest:
    """Anomaly scores in (0, 1): ~0.5 for ordinary points, near 1 for points
    isolated in very few random splits."""

    n_trees: int = 200
    sample_size: int = 256
    seed: int = 0

    def fit_score(self, x: FloatArray) -> FloatArray:
        rng = np.random.default_rng(self.seed)
        n = len(x)
        sample = min(self.sample_size, n)
        max_depth = math.ceil(math.log2(sample))
        trees = []
        for _ in range(self.n_trees):
            idx = rng.choice(n, size=sample, replace=False)
            trees.append(_grow(x[idx], 0, max_depth, rng))
        c = _average_bst_depth(sample)
        scores = np.empty(n)
        for i in range(n):
            mean_path = float(np.mean([_path_length(t, x[i], 0) for t in trees]))
            scores[i] = 2.0 ** (-mean_path / c)
        return scores


def isolation_forest_detector(y: FloatArray, n_flags: int) -> IntArray:
    """Score days by isolating (residual, day-over-day change) and flag the
    `n_flags` most isolated."""
    _, _, residual = decompose(y)
    diff = np.concatenate(([0.0], np.diff(y)))
    features = np.column_stack([residual, diff])
    scores = IsolationForest().fit_score(features)
    flagged: IntArray = np.sort(np.argsort(scores)[-n_flags:]).astype(np.int64)
    return flagged
