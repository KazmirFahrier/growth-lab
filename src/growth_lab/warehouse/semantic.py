"""Semantic metric layer.

Every ratio metric is defined as numerator-aggregate over denominator-aggregate
(ratio of sums). Averaging per-row or per-day ratios is impossible by
construction: the registry stores aggregate SQL expressions, never row math.
Unknown metrics raise KeyError — no silent fallbacks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import duckdb
import pandas as pd

MART = "marts.mart_daily_channel"
DIMENSIONS = frozenset({"channel", "date"})


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


@dataclass(frozen=True)
class MetricFilters:
    """Typed filters accepted at external query boundaries.

    Values are always passed to DuckDB as parameters. Callers cannot inject
    expressions, subqueries, or references to the sealed scoring schema.
    """

    channel: str | None = None
    start_date: date | None = None
    end_date: date | None = None

    def __post_init__(self) -> None:
        if self.channel is not None:
            if not self.channel.strip():
                raise ValueError("channel filter must not be empty")
            if len(self.channel) > 128:
                raise ValueError("channel filter is too long")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> MetricFilters:
        if not values:
            return cls()
        unknown = set(values) - {"channel", "start_date", "end_date"}
        if unknown:
            raise ValueError(f"unknown filter(s): {sorted(unknown)}")

        def parse_date(name: str) -> date | None:
            raw = values.get(name)
            if raw is None:
                return None
            if isinstance(raw, date):
                return raw
            if not isinstance(raw, str):
                raise ValueError(f"{name} must be an ISO date")
            try:
                return date.fromisoformat(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be an ISO date") from error

        channel = values.get("channel")
        if channel is not None and not isinstance(channel, str):
            raise ValueError("channel must be a string")
        return cls(
            channel=channel,
            start_date=parse_date("start_date"),
            end_date=parse_date("end_date"),
        )

    def clauses(self) -> tuple[list[str], tuple[object, ...]]:
        clauses: list[str] = []
        parameters: list[object] = []
        if self.channel is not None:
            clauses.append("channel = ?")
            parameters.append(self.channel)
        if self.start_date is not None:
            clauses.append("date >= ?")
            parameters.append(self.start_date)
        if self.end_date is not None:
            clauses.append("date <= ?")
            parameters.append(self.end_date)
        return clauses, tuple(parameters)


def _validate_dimensions(by: list[str] | None) -> list[str]:
    dims = list(by or [])
    invalid = [dimension for dimension in dims if dimension not in DIMENSIONS]
    if invalid:
        raise ValueError(f"unknown dimension(s): {invalid}; known: {sorted(DIMENSIONS)}")
    if len(dims) != len(set(dims)):
        raise ValueError("dimensions must not contain duplicates")
    return dims


def _select_clause(metrics: list[str], dims: list[str]) -> str:
    missing = [metric for metric in metrics if metric not in METRICS]
    if missing:
        raise KeyError(f"unknown metric(s): {missing}; known: {sorted(METRICS)}")
    if not metrics:
        raise ValueError("at least one metric is required")
    select = dims + [f"{METRICS[metric].sql()} AS {metric}" for metric in metrics]
    return f"SELECT {', '.join(select)} FROM {MART}"


def metric_query(
    metrics: list[str],
    by: list[str] | None = None,
    where: str | None = None,
) -> str:
    """Build the SQL for the requested metrics. Raises KeyError on unknown names."""
    dims = _validate_dimensions(by)
    sql = _select_clause(metrics, dims)
    if where:
        sql += f" WHERE {where}"
    if dims:
        sql += f" GROUP BY {', '.join(dims)} ORDER BY {', '.join(dims)}"
    return sql


def parameterized_metric_query(
    metrics: list[str],
    by: list[str] | None = None,
    filters: MetricFilters | None = None,
) -> tuple[str, tuple[object, ...]]:
    """Build a metric query whose external values are bound parameters."""
    dims = _validate_dimensions(by)
    sql = _select_clause(metrics, dims)
    clauses, parameters = (filters or MetricFilters()).clauses()
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if dims:
        sql += f" GROUP BY {', '.join(dims)} ORDER BY {', '.join(dims)}"
    return sql, parameters


def compute_metrics(
    con: duckdb.DuckDBPyConnection,
    metrics: list[str],
    by: list[str] | None = None,
    where: str | None = None,
    filters: MetricFilters | None = None,
) -> pd.DataFrame:
    """Execute a metric query against an open warehouse connection."""
    if where is not None and filters is not None:
        raise ValueError("where and filters cannot be used together")
    if filters is not None:
        sql, parameters = parameterized_metric_query(metrics, by=by, filters=filters)
        return cast(pd.DataFrame, con.execute(sql, parameters).df())
    return cast(pd.DataFrame, con.execute(metric_query(metrics, by=by, where=where)).df())
