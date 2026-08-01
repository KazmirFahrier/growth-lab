"""Typed access to the sealed ground truth (`truth.yaml`).

Only the simulator and the calibration/scoring harness may import this module.
Estimators and warehouse code are barred from it (see tests/test_no_truth_leak.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRUTH_PATH = Path(os.environ.get("GROWTH_LAB_TRUTH_PATH", str(REPO_ROOT / "truth.yaml")))


@dataclass(frozen=True)
class ChannelParams:
    """Ground-truth parameters for one paid acquisition channel."""

    name: str
    daily_spend: float
    cpm: float
    ctr_low: float
    ctr_high: float
    cvr_low: float
    cvr_high: float
    high_intent_exposure_boost: float

    def high_impression_share(self, p_high_intent: float) -> float:
        """Share of this channel's impressions served to high-intent users."""
        w_high = p_high_intent * self.high_intent_exposure_boost
        w_low = 1.0 - p_high_intent
        return w_high / (w_high + w_low)

    def expected_ctr(self, p_high_intent: float) -> float:
        """Analytic CTR implied by the parameters (mixture over intent)."""
        s = self.high_impression_share(p_high_intent)
        return s * self.ctr_high + (1.0 - s) * self.ctr_low

    def high_click_share(self, p_high_intent: float) -> float:
        """Share of this channel's *clicks* coming from high-intent users."""
        s = self.high_impression_share(p_high_intent)
        return s * self.ctr_high / self.expected_ctr(p_high_intent)

    def expected_cvr(self, p_high_intent: float) -> float:
        """Analytic click->signup CVR implied by the parameters."""
        c = self.high_click_share(p_high_intent)
        return c * self.cvr_high + (1.0 - c) * self.cvr_low


@dataclass(frozen=True)
class Truth:
    """The complete sealed data-generating process."""

    seed_default: int
    horizon_days: int
    start_date: date
    p_high_intent: float
    channels: tuple[ChannelParams, ...]
    organic_daily_signups_low: float
    organic_daily_signups_high: float
    dow_multipliers: tuple[float, ...]
    trial_to_paid_low: float
    trial_to_paid_high: float
    plan_pro_share_low: float
    plan_pro_share_high: float
    price_basic: float
    price_pro: float
    monthly_hazard_basic: float
    monthly_hazard_pro: float
    txn_fraud_rate: float

    def channel(self, name: str) -> ChannelParams:
        for ch in self.channels:
            if ch.name == name:
                return ch
        raise KeyError(f"unknown channel: {name!r}")


def load_truth(path: Path | None = None) -> Truth:
    """Load and validate the sealed ground truth. Fails loudly on any anomaly."""
    truth_path = path if path is not None else DEFAULT_TRUTH_PATH
    with truth_path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    channels = tuple(
        ChannelParams(name=name, **{str(k): float(v) for k, v in cfg.items()})
        for name, cfg in raw["channels"].items()
    )
    dow = tuple(float(x) for x in raw["seasonality"]["dow_multipliers"])
    if len(dow) != 7:
        raise ValueError(f"dow_multipliers must have 7 entries, got {len(dow)}")

    truth = Truth(
        seed_default=int(raw["seed_default"]),
        horizon_days=int(raw["horizon_days"]),
        start_date=raw["start_date"],
        p_high_intent=float(raw["population"]["p_high_intent"]),
        channels=channels,
        organic_daily_signups_low=float(raw["organic"]["daily_signups_low"]),
        organic_daily_signups_high=float(raw["organic"]["daily_signups_high"]),
        dow_multipliers=dow,
        trial_to_paid_low=float(raw["subscription"]["trial_to_paid"]["low"]),
        trial_to_paid_high=float(raw["subscription"]["trial_to_paid"]["high"]),
        plan_pro_share_low=float(raw["subscription"]["plan_pro_share"]["low"]),
        plan_pro_share_high=float(raw["subscription"]["plan_pro_share"]["high"]),
        price_basic=float(raw["subscription"]["price_basic"]),
        price_pro=float(raw["subscription"]["price_pro"]),
        monthly_hazard_basic=float(raw["churn"]["monthly_hazard_basic"]),
        monthly_hazard_pro=float(raw["churn"]["monthly_hazard_pro"]),
        txn_fraud_rate=float(raw["fraud"]["txn_fraud_rate"]),
    )

    for p in (
        truth.p_high_intent,
        truth.trial_to_paid_low,
        truth.trial_to_paid_high,
        truth.plan_pro_share_low,
        truth.plan_pro_share_high,
        truth.monthly_hazard_basic,
        truth.monthly_hazard_pro,
        truth.txn_fraud_rate,
        *(x for ch in truth.channels for x in (ch.ctr_low, ch.ctr_high, ch.cvr_low, ch.cvr_high)),
    ):
        if not 0.0 < p < 1.0:
            raise ValueError(f"probability parameter out of (0, 1): {p}")
    if truth.horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    return truth
