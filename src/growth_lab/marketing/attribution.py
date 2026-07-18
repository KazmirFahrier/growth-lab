"""Multi-touch attribution: last-touch, linear, and Markov removal effect.

Attribution models divide observed conversions among channels; none of them
measures incrementality. The recovery tests quantify exactly how far each
model strays from ground truth — last-touch hands the retargeting channel
credit for conversions that were already going to happen.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]

START, CONVERTED, NULL = "_start", "_converted", "_null"


def last_touch(paths: list[list[str]], converted: BoolArray) -> dict[str, float]:
    """All credit to the final touchpoint before conversion."""
    credit: dict[str, float] = defaultdict(float)
    for path, conv in zip(paths, converted, strict=True):
        if conv and path:
            credit[path[-1]] += 1.0
    return dict(credit)


def linear_touch(paths: list[list[str]], converted: BoolArray) -> dict[str, float]:
    """Equal credit to every touchpoint on a converting path."""
    credit: dict[str, float] = defaultdict(float)
    for path, conv in zip(paths, converted, strict=True):
        if conv and path:
            share = 1.0 / len(path)
            for touch in path:
                credit[touch] += share
    return dict(credit)


def _transition_counts(
    paths: list[list[str]], converted: BoolArray
) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for path, conv in zip(paths, converted, strict=True):
        states = [START, *path, CONVERTED if conv else NULL]
        for a, b in itertools.pairwise(states):
            counts[a][b] += 1.0
    return counts


def _conversion_probability(
    counts: dict[str, dict[str, float]], removed: str | None = None
) -> float:
    """Absorption probability into CONVERTED for the first-order chain,
    with `removed` (if given) treated as a NULL absorber."""
    states = sorted({s for s in counts} | {t for row in counts.values() for t in row})
    transient = [s for s in states if s not in (CONVERTED, NULL) and s != removed]
    index = {s: i for i, s in enumerate(transient)}
    n = len(transient)
    q = np.zeros((n, n))  # transient -> transient
    r = np.zeros(n)  # transient -> CONVERTED
    for s in transient:
        row = counts.get(s, {})
        total = sum(row.values())
        if total == 0:
            continue
        for target, c in row.items():
            p = c / total
            if target in (removed, NULL):
                continue  # absorbed without converting
            elif target == CONVERTED:
                r[index[s]] += p
            elif target in index:
                q[index[s], index[target]] += p
    absorbed = np.linalg.solve(np.eye(n) - q, r)
    return float(absorbed[index[START]])


def markov_removal(
    paths: list[list[str]], converted: BoolArray, channels: tuple[str, ...]
) -> dict[str, float]:
    """Removal-effect attribution on a first-order Markov chain.

    Each channel's effect is the drop in the chain's conversion probability
    when that channel is removed; credits are normalized to total observed
    conversions.
    """
    counts = _transition_counts(paths, converted)
    p_full = _conversion_probability(counts)
    if p_full <= 0:
        raise ValueError("chain never converts; attribution is undefined")
    removal = {
        ch: max(p_full - _conversion_probability(counts, removed=ch), 0.0) for ch in channels
    }
    total_removal = sum(removal.values())
    if total_removal == 0:
        raise ValueError("no channel affects conversion in the fitted chain")
    total_conversions = float(np.asarray(converted).sum())
    return {ch: total_conversions * eff / total_removal for ch, eff in removal.items()}
