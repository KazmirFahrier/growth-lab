"""Probability quality: AUC, Brier, reliability, ECE, cost-optimal thresholds.

A fraud model that ranks well but reports miscalibrated probabilities will
set wrong review thresholds and misprice risk. Calibration is a deployment
gate here, not a nice-to-have.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def auc(labels: BoolArray, scores: FloatArray) -> float:
    """Mann-Whitney rank AUC (tie-aware)."""
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need both classes to compute AUC")
    order = np.argsort(np.concatenate([neg, pos]), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties
    combined = np.concatenate([neg, pos])
    for value in np.unique(combined):
        mask = combined == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    rank_sum_pos = float(ranks[len(neg) :].sum())
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def brier_score(labels: BoolArray, probabilities: FloatArray) -> float:
    return float(np.mean((probabilities - labels.astype(np.float64)) ** 2))


@dataclass(frozen=True)
class ReliabilityBin:
    mean_predicted: float
    observed_rate: float
    count: int


def reliability_curve(
    labels: BoolArray, probabilities: FloatArray, n_bins: int = 10
) -> tuple[ReliabilityBin, ...]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in itertools.pairwise(edges):
        mask = (probabilities >= lo) & (probabilities < hi)
        if not mask.any():
            continue
        bins.append(
            ReliabilityBin(
                mean_predicted=float(probabilities[mask].mean()),
                observed_rate=float(labels[mask].mean()),
                count=int(mask.sum()),
            )
        )
    return tuple(bins)


def expected_calibration_error(
    labels: BoolArray, probabilities: FloatArray, n_bins: int = 10
) -> float:
    bins = reliability_curve(labels, probabilities, n_bins)
    total = sum(b.count for b in bins)
    return sum(
        (b.count / total) * abs(b.observed_rate - b.mean_predicted) for b in bins
    )


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    expected_cost_per_case: float
    false_positive_cost: float
    false_negative_cost: float


def cost_optimal_threshold(
    labels: BoolArray,
    probabilities: FloatArray,
    false_positive_cost: float,
    false_negative_cost: float,
) -> ThresholdChoice:
    """Pick the flagging threshold that minimizes empirical expected cost.

    For a perfectly calibrated model this lands at c_fp / (c_fp + c_fn) —
    the recovery test checks exactly that.
    """
    if false_positive_cost <= 0 or false_negative_cost <= 0:
        raise ValueError("costs must be positive")
    candidates = np.unique(np.concatenate([[0.0, 1.0], np.round(probabilities, 4)]))
    y = labels.astype(bool)
    best_threshold, best_cost = 0.5, np.inf
    for threshold in candidates:
        flagged = probabilities >= threshold
        cost = (
            false_positive_cost * float((flagged & ~y).sum())
            + false_negative_cost * float((~flagged & y).sum())
        ) / len(y)
        if cost < best_cost:
            best_cost, best_threshold = cost, float(threshold)
    return ThresholdChoice(
        threshold=best_threshold,
        expected_cost_per_case=float(best_cost),
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
