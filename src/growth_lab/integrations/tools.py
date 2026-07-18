"""growth-lab's models and warehouse, exposed as agent-callable tools.

These are the tools campaign-copilot's agent loop can mount to answer
questions like "should we shift budget to channel B?" with numbers that are
measured, not asserted: every value in a tool's `content` also appears in
its `data`, so the grounding checker can license each claim.

Failures are results with machine-readable codes, never exceptions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from growth_lab.forecasting import HoltWinters
from growth_lab.integrations.contract import ToolResult, ToolSpec
from growth_lab.marketing import ChannelResponse, fit_geometric_ltv, optimal_allocation
from growth_lab.warehouse.semantic import METRICS, compute_metrics, metric_query

_FORBIDDEN_FILTER = re.compile(r";|--|\b(drop|delete|insert|update|attach|copy)\b", re.IGNORECASE)


def _connect(db_path: Path) -> duckdb.DuckDBPyConnection | ToolResult:
    if not db_path.exists():
        return ToolResult.failure(
            "WAREHOUSE_UNAVAILABLE",
            f"no warehouse at {db_path}; run `python -m growth_lab build` first",
        )
    return duckdb.connect(str(db_path), read_only=True)


@dataclass
class QueryGrowthMetricsTool:
    """Semantic-layer metrics: ratio-of-sums by construction."""

    db_path: Path

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="query_growth_metrics",
            description=(
                "Compute governed growth metrics (ratio-of-sums semantic layer). "
                f"Known metrics: {', '.join(sorted(METRICS))}. Optional grouping "
                "by 'channel' or 'date' and a simple SQL filter on those columns."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "by": {"type": "array", "items": {"type": "string"}},
                    "where": {"type": "string"},
                },
                "required": ["metrics"],
            },
        )

    def run(self, **kwargs: Any) -> ToolResult:
        metrics = list(kwargs.get("metrics", []))
        by = list(kwargs.get("by", []) or [])
        where = kwargs.get("where")
        unknown = [m for m in metrics if m not in METRICS]
        if not metrics or unknown:
            return ToolResult.failure(
                "INVALID_METRIC",
                f"unknown metric(s) {unknown}; known: {sorted(METRICS)}",
            )
        if where and _FORBIDDEN_FILTER.search(where):
            return ToolResult.failure(
                "FORBIDDEN_SQL", "filter may only restrict rows, not run statements"
            )
        con = _connect(self.db_path)
        if isinstance(con, ToolResult):
            return con
        try:
            frame = compute_metrics(con, metrics, by=by or None, where=where)
        finally:
            con.close()
        rows = frame.to_dict(orient="records")
        return ToolResult.success(
            f"{len(rows)} row(s) for {', '.join(metrics)}"
            + (f" by {', '.join(by)}" if by else ""),
            row_count=len(rows),
            rows=rows,
            sql=metric_query(metrics, by=by or None, where=where),
        )


@dataclass
class LtvSummaryTool:
    """Censored-geometric LTV by plan and acquisition channel."""

    db_path: Path

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ltv_summary",
            description=(
                "Customer lifetime value from billing history: censored-geometric "
                "churn MLE per plan, plus average LTV by acquisition channel."
            ),
            input_schema={"type": "object", "properties": {}},
        )

    def run(self, **kwargs: Any) -> ToolResult:
        con = _connect(self.db_path)
        if isinstance(con, ToolResult):
            return con
        try:
            import pandas as pd

            users = con.execute(
                "SELECT user_id, signup_date, channel, is_paid AS subscribed, plan "
                "FROM marts.dim_users"
            ).df()
            txns = con.execute(
                "SELECT txn_id, user_id, txn_date, amount, is_fraud "
                "FROM marts.fct_transactions"
            ).df()
            row = con.execute("SELECT max(date) FROM marts.mart_daily_channel").fetchone()
        finally:
            con.close()
        if row is None or row[0] is None:
            return ToolResult.failure("WAREHOUSE_EMPTY", "no dated data in the warehouse")
        estimate = fit_geometric_ltv(users, txns, pd.Timestamp(row[0]))
        plans = [
            {
                "plan": p.plan,
                "monthly_churn": round(p.monthly_churn, 4),
                "ltv": round(p.ltv, 2),
            }
            for p in estimate.plans
        ]
        return ToolResult.success(
            f"LTV estimated for {len(plans)} plan(s)",
            plans=plans,
            by_channel=estimate.by_channel.round(2).to_dict(orient="records"),
        )


@dataclass
class ForecastRevenueTool:
    """Holt-Winters revenue outlook."""

    db_path: Path
    max_horizon: int = 56

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="forecast_revenue",
            description="Forecast total daily revenue with additive Holt-Winters.",
            input_schema={
                "type": "object",
                "properties": {
                    "horizon_days": {"type": "integer", "minimum": 1, "maximum": self.max_horizon}
                },
                "required": ["horizon_days"],
            },
        )

    def run(self, **kwargs: Any) -> ToolResult:
        horizon = int(kwargs.get("horizon_days", 0))
        if not 1 <= horizon <= self.max_horizon:
            return ToolResult.failure(
                "INVALID_HORIZON", f"horizon_days must be in [1, {self.max_horizon}]"
            )
        con = _connect(self.db_path)
        if isinstance(con, ToolResult):
            return con
        try:
            series = con.execute(
                "SELECT date, SUM(revenue) AS revenue FROM marts.mart_daily_channel "
                "GROUP BY date ORDER BY date"
            ).df()
        finally:
            con.close()
        model = HoltWinters()
        model.fit(series["revenue"].to_numpy(dtype=np.float64))
        point = model.predict(horizon)
        return ToolResult.success(
            f"forecast for the next {horizon} day(s)",
            horizon_days=horizon,
            total=round(float(point.sum()), 2),
            daily=[round(float(v), 2) for v in point],
        )


@dataclass
class BudgetPlannerTool:
    """Optimal budget split over MMM response curves — the bridge's demo:
    'should we shift budget to channel B?' answered from measured curves."""

    params_path: Path

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="plan_budget",
            description=(
                "Allocate a total daily media budget across channels to maximize "
                "steady-state revenue, using fitted MMM response curves "
                "(adstock + saturation). Compares against the current allocation."
            ),
            input_schema={
                "type": "object",
                "properties": {"total_daily_budget": {"type": "number", "minimum": 1}},
                "required": ["total_daily_budget"],
            },
        )

    def run(self, **kwargs: Any) -> ToolResult:
        budget = float(kwargs.get("total_daily_budget", 0.0))
        if budget <= 0:
            return ToolResult.failure("INVALID_BUDGET", "total_daily_budget must be positive")
        if not self.params_path.exists():
            return ToolResult.failure(
                "NO_MMM_PARAMS",
                f"no fitted MMM parameters at {self.params_path}; "
                "run `python -m growth_lab export-mmm` first",
            )
        params = json.loads(self.params_path.read_text())
        responses = tuple(
            ChannelResponse(c["name"], c["beta"], c["decay"], c["half_sat"])
            for c in params["channels"]
        )
        current = {c["name"]: float(c["current_daily_spend"]) for c in params["channels"]}

        allocation = optimal_allocation(responses, budget)
        current_revenue = sum(r.revenue(current[r.name]) for r in responses)
        shifts: list[dict[str, str | float]] = [
            {
                "channel": name,
                "current": round(current[name], 2),
                "recommended": round(float(spend), 2),
                "change": round(float(spend) - current[name], 2),
            }
            for name, spend in zip(allocation.channels, allocation.daily_spend, strict=True)
        ]
        uplift = allocation.expected_daily_revenue - current_revenue
        biggest = max(shifts, key=lambda s: abs(float(str(s["change"]))))
        return ToolResult.success(
            f"reallocating moves expected daily revenue by {uplift:+,.2f}; "
            f"largest shift: {biggest['channel']} {biggest['change']:+,.2f}",
            total_daily_budget=budget,
            shifts=shifts,
            expected_daily_revenue_current=round(current_revenue, 2),
            expected_daily_revenue_optimal=round(allocation.expected_daily_revenue, 2),
            expected_daily_uplift=round(uplift, 2),
            source=params.get("source", "fitted MMM parameters"),
        )


def default_toolkit(db_path: Path, mmm_params_path: Path) -> tuple[Any, ...]:
    """The set campaign-copilot mounts."""
    return (
        QueryGrowthMetricsTool(db_path),
        LtvSummaryTool(db_path),
        ForecastRevenueTool(db_path),
        BudgetPlannerTool(mmm_params_path),
    )
