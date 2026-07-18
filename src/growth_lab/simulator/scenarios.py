"""Causal scenario bank: situations engineered to break naive analysis.

Each scenario returns its data *and* its ground-truth effect. The truth field
is for the scoring harness only — the sealed `causal` package never imports
this module (enforced by tests/test_no_truth_leak.py).

Every scenario has a `broken` variant that violates the identifying
assumption; estimators are required to refuse those datasets, not to produce
a confident wrong answer on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


# --- DiD: staggered feature rollout by geo ---------------------------------

@dataclass(frozen=True)
class GeoRolloutScenario:
    panel: pd.DataFrame  # geo, day, treated, post, y
    true_att: float


def geo_rollout(
    seed: int = 0,
    n_geos: int = 60,
    n_days: int = 120,
    rollout_day: int = 60,
    effect: float = 0.008,
    differential_trends: bool = False,
) -> GeoRolloutScenario:
    """Feature rollout to the *strongest* geos (deliberate selection bias).

    Treated geos have higher baselines, so a post-period cross-section
    comparison overstates the effect badly. Shared day shocks confound
    pre/post comparisons. DiD identifies the effect — unless
    `differential_trends` is set, which breaks parallel trends and must be
    caught by the estimator's placebo check.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.05, 0.012, size=n_geos).clip(0.01, 0.2)
    # top half of geos by baseline get the feature first: selection on levels
    treated = np.zeros(n_geos, dtype=bool)
    treated[np.argsort(base)[n_geos // 2 :]] = True

    days = np.arange(n_days)
    day_shock = 0.004 * np.sin(2 * math.pi * days / 7) + rng.normal(0, 0.0015, size=n_days)

    geo_idx = np.repeat(np.arange(n_geos), n_days)
    day_idx = np.tile(days, n_geos)
    y = (
        base[geo_idx]
        + day_shock[day_idx]
        + effect * (treated[geo_idx] & (day_idx >= rollout_day))
        + rng.normal(0, 0.003, size=n_geos * n_days)
    )
    if differential_trends:
        y = y + 0.00008 * day_idx * treated[geo_idx]

    panel = pd.DataFrame(
        {
            "geo": geo_idx,
            "day": day_idx,
            "treated": treated[geo_idx],
            "post": day_idx >= rollout_day,
            "y": y,
        }
    )
    return GeoRolloutScenario(panel=panel, true_att=effect)


# --- RDD: benefit granted above a spend threshold ---------------------------

@dataclass(frozen=True)
class SpendThresholdScenario:
    running: FloatArray  # past-spend score (running variable)
    outcome: FloatArray
    cutoff: float
    true_effect: float


def spend_threshold(
    seed: int = 1,
    n: int = 8000,
    cutoff: float = 0.3,
    effect: float = 0.15,
    manipulated: bool = False,
) -> SpendThresholdScenario:
    """A perk kicks in above a spend score cutoff.

    Outcome rises smoothly with the score, so above/below means differ far
    more than the true effect. Sharp RDD identifies it at the cutoff. The
    `manipulated` variant bunches users just past the threshold (gaming),
    which must trip the density-continuity check.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n)
    if manipulated:
        just_below = (x < cutoff) & (x > cutoff - 0.1)
        bump = just_below & (rng.random(n) < 0.6)
        x = np.where(bump, x + 0.12, x)
    d = x >= cutoff
    y = 0.4 + 0.35 * x - 0.08 * x**2 + effect * d + rng.normal(0, 0.15, size=n)
    return SpendThresholdScenario(running=x, outcome=y, cutoff=cutoff, true_effect=effect)


# --- IPW / uplift: self-selected promo email --------------------------------

@dataclass(frozen=True)
class PromoEmailScenario:
    covariates: FloatArray  # columns: engagement, tenure
    treated: BoolArray
    outcome: FloatArray
    true_ate: float
    true_cate: FloatArray


def promo_email(
    seed: int = 2,
    n: int = 6000,
    heterogeneous: bool = False,
    no_overlap: bool = False,
) -> PromoEmailScenario:
    """A promo that engaged users are more likely to receive.

    Engagement drives both exposure and baseline conversion, so the raw
    treated-vs-control difference overstates the effect. IPW on the observed
    covariates recovers it. With `heterogeneous`, the effect concentrates in
    low-engagement users (for uplift models to find). With `no_overlap`,
    exposure is nearly deterministic in engagement and IPW must refuse.
    """
    rng = np.random.default_rng(seed)
    engagement = rng.normal(0.0, 1.0, size=n)
    tenure = rng.normal(0.0, 1.0, size=n)
    covariates = np.column_stack([engagement, tenure])

    if no_overlap:
        propensity = 1.0 / (1.0 + np.exp(-6.0 * engagement))
    else:
        propensity = 1.0 / (1.0 + np.exp(-(0.8 * engagement + 0.4 * tenure - 0.3)))
    treated = rng.random(n) < propensity

    p_base = 1.0 / (1.0 + np.exp(-(-1.0 + 0.9 * engagement + 0.5 * tenure)))
    tau = np.where(engagement < 0.0, 0.10, 0.01) if heterogeneous else np.full(n, 0.06)
    p_treated = np.minimum(p_base + tau, 0.98)
    true_cate = p_treated - p_base

    p_actual = np.where(treated, p_treated, p_base)
    outcome = (rng.random(n) < p_actual).astype(np.float64)
    return PromoEmailScenario(
        covariates=covariates,
        treated=treated,
        outcome=outcome,
        true_ate=float(true_cate.mean()),
        true_cate=true_cate,
    )


# --- IV: endogenous price with a cost-shifter instrument --------------------

@dataclass(frozen=True)
class PriceInstrumentScenario:
    instrument: FloatArray  # cost shock (excluded)
    price: FloatArray
    demand: FloatArray
    true_price_coefficient: float


def price_instrument(
    seed: int = 3,
    n: int = 6000,
    price_coefficient: float = -0.4,
    weak: bool = False,
) -> PriceInstrumentScenario:
    """Price is set knowing demand, so OLS on (price, demand) is wildly
    biased — here the bias flips the sign. A cost shock that moves price but
    not demand identifies the true coefficient via 2SLS. The `weak` variant
    makes the instrument nearly irrelevant and must trip the first-stage
    F check.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, size=n)  # cost shock
    u = rng.normal(0.0, 1.0, size=n)  # unobserved demand shock
    pi = 0.01 if weak else 0.5
    price = 10.0 + pi * z + 0.6 * u + rng.normal(0, 0.3, size=n)
    demand = 5.0 + price_coefficient * price + 2.0 * u + rng.normal(0, 0.5, size=n)
    return PriceInstrumentScenario(
        instrument=z,
        price=price,
        demand=demand,
        true_price_coefficient=price_coefficient,
    )


# --- MMM: media market with adstock + saturation ----------------------------

MMM_CHANNELS = ("search", "social", "display", "video")


@dataclass(frozen=True)
class MmmScenario:
    spend: FloatArray  # (n_days, n_channels)
    revenue: FloatArray
    day_of_week: npt.NDArray[np.int64]
    channels: tuple[str, ...]
    true_beta: FloatArray  # revenue at full saturation, per channel
    true_decay: FloatArray  # geometric adstock carryover
    true_half_sat: FloatArray  # spend level of half saturation (adstocked)
    true_base: float
    true_noise_sd: float
    true_roas: FloatArray  # incremental revenue / spend, per channel
    true_contribution: FloatArray  # (n_days, n_channels) incremental revenue


def _adstock(spend: FloatArray, decay: float) -> FloatArray:
    out = np.empty_like(spend)
    carry = 0.0
    for t in range(len(spend)):
        carry = spend[t] + decay * carry
        out[t] = carry
    return out


def mmm_market(seed: int = 4, n_days: int = 600) -> MmmScenario:
    """Four media channels driving revenue through carryover and saturation.

    Spend follows independent AR(1)-in-log dynamics with campaign pulses and
    several 14-day blackout windows per channel (go-dark periods) — the
    variation that makes carryover and saturation identifiable in practice.
    The true transform per channel is geometric adstock followed by a
    Michaelis-Menten saturation curve; the scenario exposes the exact per-day
    incremental contribution so MMM estimates can be scored against truth.
    """
    rng = np.random.default_rng(seed)
    base_spend = np.array([800.0, 500.0, 300.0, 200.0])
    beta = np.array([4000.0, 2500.0, 1200.0, 800.0])
    decay = np.array([0.2, 0.5, 0.7, 0.4])
    half_sat = np.array([900.0, 600.0, 400.0, 300.0])
    true_base = 5000.0
    noise_sd = 250.0
    dow_effect = np.array([0.0, 50.0, 100.0, 150.0, 300.0, -200.0, -300.0])

    days = np.arange(n_days)
    n_ch = len(MMM_CHANNELS)
    spend = np.empty((n_days, n_ch))
    for c in range(n_ch):
        ar = np.empty(n_days)
        level = 0.0
        for t in range(n_days):
            level = 0.8 * level + rng.normal(0, 0.25)
            ar[t] = level
        pulse = np.where(rng.random(n_days) < 0.15, 2.0, 1.0)
        dark = np.ones(n_days)
        for _ in range(4):  # go-dark windows: strongest identification signal
            start = int(rng.integers(0, n_days - 14))
            dark[start : start + 14] = 0.0
        spend[:, c] = base_spend[c] * np.exp(ar) * pulse * dark

    contribution = np.empty((n_days, n_ch))
    for c in range(n_ch):
        adstocked = _adstock(spend[:, c], float(decay[c]))
        contribution[:, c] = beta[c] * adstocked / (adstocked + half_sat[c])

    dow = (days % 7).astype(np.int64)
    revenue = (
        true_base
        + dow_effect[dow]
        + contribution.sum(axis=1)
        + rng.normal(0, noise_sd, size=n_days)
    )
    true_roas = contribution.sum(axis=0) / spend.sum(axis=0)
    return MmmScenario(
        spend=spend,
        revenue=revenue,
        day_of_week=dow,
        channels=MMM_CHANNELS,
        true_beta=beta,
        true_decay=decay,
        true_half_sat=half_sat,
        true_base=true_base,
        true_noise_sd=noise_sd,
        true_roas=true_roas,
        true_contribution=contribution,
    )


# --- attribution: multi-touch journeys with a retargeting harvester ---------

@dataclass(frozen=True)
class JourneyScenario:
    paths: list[list[str]]  # touchpoint sequences, converters and not
    converted: BoolArray
    channels: tuple[str, ...]
    true_incremental: dict[str, float]  # conversions each channel truly caused


def journey_paths(seed: int = 5, n_users: int = 20_000) -> JourneyScenario:
    """User journeys where display retargets users already about to convert.

    Conversion probability is base(intent) plus a small additive lift per
    touch. Display's lift is near zero, but it is inserted late into
    high-intent journeys — so last-touch attribution hands it credit for
    conversions that were already going to happen. With additive lifts the
    true incremental contribution of a channel is exactly the sum of its
    per-touch lifts over users, which is what estimators are scored against.
    """
    rng = np.random.default_rng(seed)
    channels = ("search", "social", "display", "video")
    lift = {"search": 0.06, "social": 0.03, "display": 0.005, "video": 0.02}
    base_by_intent = {True: 0.25, False: 0.03}
    # organic-ish channel mix for the journey body (no display: it only retargets)
    body_channels = ("search", "social", "video")
    body_probs = (0.5, 0.3, 0.2)

    paths: list[list[str]] = []
    converted = np.zeros(n_users, dtype=bool)
    true_incremental = dict.fromkeys(channels, 0.0)

    high = rng.random(n_users) < 0.25
    n_touches = rng.poisson(1.6, size=n_users) + 1
    for i in range(n_users):
        path = list(rng.choice(body_channels, size=n_touches[i], p=body_probs))
        # retargeting: display finds high-intent users and appends itself late
        if high[i] and rng.random() < 0.8:
            path.append("display")
        p_conv = base_by_intent[bool(high[i])]
        for touch in path:
            p_conv += lift[touch]
            true_incremental[touch] += lift[touch]
        converted[i] = rng.random() < min(p_conv, 0.95)
        paths.append(path)
    return JourneyScenario(
        paths=paths,
        converted=converted,
        channels=channels,
        true_incremental=true_incremental,
    )
