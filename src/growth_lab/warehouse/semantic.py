"""Semantic metric layer.

Every ratio metric is defined as numerator-aggregate over denominator-aggregate
(ratio of sums). Averaging per-row or per-day ratios is impossible by
construction: the registry stores aggregate SQL expressions, never row math.
Unknown metrics raise KeyError — no silent fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import duckdb
import pandas as pd

MART = "marts.mart_daily_channel"


@dataclass(frozen=True)
class Metric:
    """A metric over the daily channel mart."""

    name: str
    numerator: str
    denominator: str | None
    description: str

    def sql(self) -> str:
        if self.denominator is None:
            return self.numerator
        return f"CAST({self.numerator} AS DOUBLE) / NULLIF({self.denominator}, 0)"


METRICS: dict[str, Metric] = {
    m.name: m
    for m in (
        Metric("spend", "SUM(spend)", None, "Total ad spend"),
        Metric("impressions", "SUM(impressions)", None, "Ad impressions delivered"),
        Metric("clicks", "SUM(clicks)", None, "Ad clicks"),
        Metric("signups", "SUM(signups)", None, "Signups (trials started)"),
        Metric("paid_signups", "SUM(paid_signups)", None, "Signups that converted to paid"),
        Metric("txns", "SUM(txns)", None, "Billing transactions"),
        Metric("revenue", "SUM(revenue)", None, "Billed revenue (acquisition-channel attributed)"),
        Metric("fraud_txns", "SUM(fraud_txns)", None, "Transactions flagged fraudulent"),
        Metric("ctr", "SUM(clicks)", "SUM(impressions)", "Click-through rate"),
        Metric("cvr", "SUM(signups)", "SUM(clicks)", "Click -> signup conversion rate"),
        Metric("cpc", "SUM(spend)", "SUM(clicks)", "Cost per click"),
        Metric("cac", "SUM(spend)", "SUM(paid_signups)", "Customer acquisition cost"),
        Metric("trial_to_paid", "SUM(paid_signups)", "SUM(signups)", "Trial -> paid rate"),
        Metric("fraud_rate", "SUM(fraud_txns)", "SUM(txns)", "Share of transactions fraudulent"),
        Metric("arpu", "SUM(revenue)", "SUM(paid_signups)", "Avg billed revenue per paid user"),
    )
}


def metric_query(
    metrics: list[str],
    by: list[str] | None = None,
    where: str | None = None,
) -> str:
    """Build the SQL for the requested metrics. Raises KeyError on unknown names."""
    missing = [m for m in metrics if m not in METRICS]
    if missing:
        raise KeyError(f"unknown metric(s): {missing}; known: {sorted(METRICS)}")
    dims = list(by or [])
    select = dims + [f"{METRICS[m].sql()} AS {m}" for m in metrics]
    sql = f"SELECT {', '.join(select)} FROM {MART}"
    if where:
        sql += f" WHERE {where}"
    if dims:
        sql += f" GROUP BY {', '.join(dims)} ORDER BY {', '.join(dims)}"
    return sql


def compute_metrics(
    con: duckdb.DuckDBPyConnection,
    metrics: list[str],
    by: list[str] | None = None,
    where: str | None = None,
) -> pd.DataFrame:
    """Execute a metric query against an open warehouse connection."""
    return cast(pd.DataFrame, con.execute(metric_query(metrics, by=by, where=where)).df())
