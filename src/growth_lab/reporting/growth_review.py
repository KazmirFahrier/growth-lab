"""The weekly growth review: warehouse in, decision memo out.

Every number is a Figure with provenance — KPI scalars carry the exact
semantic-layer SQL that produced them, model outputs carry their estimator
lineage. The memo template itself contains no digits; the provenance gate
enforces that at test time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from growth_lab.forecasting import HoltWinters
from growth_lab.marketing import fit_geometric_ltv
from growth_lab.reporting.provenance import Figure, FigureRegistry
from growth_lab.risk import mad_residual_detector
from growth_lab.warehouse.semantic import compute_metrics, metric_query

KPI_METRICS = ("spend", "signups", "paid_signups", "revenue", "cac", "fraud_rate")
KPI_UNITS = {
    "spend": "$",
    "revenue": "$",
    "cac": "$",
    "fraud_rate": "%",
    "signups": "int",
    "paid_signups": "int",
}
FORECAST_HORIZON_DAYS = 14


@dataclass(frozen=True)
class GrowthReview:
    as_of: date
    markdown: str
    figures: FigureRegistry


def _kpi_figures(
    con: duckdb.DuckDBPyConnection, figures: FigureRegistry, week_start: date, as_of: date
) -> None:
    prior_start = week_start - timedelta(days=7)
    this_where = f"date >= '{week_start}'"
    prior_where = f"date >= '{prior_start}' AND date < '{week_start}'"

    this_week = compute_metrics(con, list(KPI_METRICS), where=this_where)
    prior_week = compute_metrics(con, list(KPI_METRICS), where=prior_where)
    for metric in KPI_METRICS:
        value = float(this_week[metric].iloc[0])
        prior = float(prior_week[metric].iloc[0])
        figures.add(
            Figure(
                slug=f"kpi_{metric}",
                title=f"{metric} (trailing week)",
                value=value,
                unit=KPI_UNITS[metric],
                source=metric_query([metric], where=this_where),
            )
        )
        figures.add(
            Figure(
                slug=f"kpi_{metric}_wow",
                title=f"{metric} week-over-week change",
                value=(value - prior) / prior if prior != 0 else 0.0,
                unit="%",
                source=(
                    f"({metric_query([metric], where=this_where)}) vs "
                    f"({metric_query([metric], where=prior_where)})"
                ),
            )
        )


def _channel_figure(
    con: duckdb.DuckDBPyConnection, figures: FigureRegistry, week_start: date
) -> None:
    where = f"date >= '{week_start}'"
    frame = compute_metrics(
        con, ["spend", "signups", "paid_signups", "cac", "revenue"], by=["channel"], where=where
    )
    figures.add(
        Figure(
            slug="channel_week",
            title="Channel performance (trailing week)",
            value=frame,
            source=metric_query(
                ["spend", "signups", "paid_signups", "cac", "revenue"],
                by=["channel"],
                where=where,
            ),
        )
    )


LTV_USERS_SQL = (
    "SELECT user_id, signup_date, channel, is_paid AS subscribed, plan FROM marts.dim_users"
)
LTV_TXNS_SQL = "SELECT txn_id, user_id, txn_date, amount, is_fraud FROM marts.fct_transactions"


def _ltv_figures(con: duckdb.DuckDBPyConnection, figures: FigureRegistry, as_of: date) -> None:
    users = con.execute(LTV_USERS_SQL).df()
    txns = con.execute(LTV_TXNS_SQL).df()
    estimate = fit_geometric_ltv(users, txns, pd.Timestamp(as_of))
    plans = pd.DataFrame(
        [
            {
                "plan": p.plan,
                "subscribers": p.n_subscribers,
                "monthly_churn": p.monthly_churn,
                "exp_lifetime_mo": p.expected_lifetime_months,
                "ltv": p.ltv,
            }
            for p in estimate.plans
        ]
    )
    lineage = (
        "censored-geometric churn MLE (growth_lab.marketing.ltv) over "
        f"[{LTV_USERS_SQL}] and [{LTV_TXNS_SQL}], censored at {as_of}"
    )
    figures.add(Figure(slug="ltv_plans", title="LTV by plan", value=plans, source=lineage))
    figures.add(
        Figure(
            slug="ltv_channels",
            title="Average LTV by acquisition channel",
            value=estimate.by_channel,
            source=lineage,
        )
    )


REVENUE_SERIES_SQL = (
    "SELECT date, SUM(revenue) AS revenue FROM marts.mart_daily_channel GROUP BY date ORDER BY date"
)


def _forecast_and_anomaly_figures(con: duckdb.DuckDBPyConnection, figures: FigureRegistry) -> None:
    series = con.execute(REVENUE_SERIES_SQL).df()
    revenue = series["revenue"].to_numpy(dtype=np.float64)

    model = HoltWinters()
    model.fit(revenue)
    point = model.predict(FORECAST_HORIZON_DAYS)
    figures.add(
        Figure(
            slug="forecast_total",
            title="Expected revenue, next two weeks",
            value=float(point.sum()),
            unit="$",
            source=(
                "Holt-Winters additive weekly forecast "
                f"(growth_lab.forecasting) fit on [{REVENUE_SERIES_SQL}]"
            ),
        )
    )
    figures.add(
        Figure(
            slug="forecast_horizon",
            title="Forecast horizon in days",
            value=float(FORECAST_HORIZON_DAYS),
            unit="int",
            source="growth_lab.reporting.growth_review.FORECAST_HORIZON_DAYS",
        )
    )

    detector = mad_residual_detector(revenue)
    figures.add(
        Figure(
            slug="anomaly_count",
            title="Anomalous revenue days detected",
            value=float(len(detector.flagged_days)),
            unit="int",
            source=(
                "MAD residual detector (growth_lab.risk.anomaly), robust "
                f"threshold {detector.threshold_sigmas:g} sigmas, on "
                f"[{REVENUE_SERIES_SQL}]"
            ),
        )
    )


def _render(figures: FigureRegistry, as_of: date, week_start: date) -> str:
    f = figures.get
    lines = [
        f"# Meridian growth review — week of {week_start} (data through {as_of})",
        "",
        "## KPIs, trailing week",
        "",
        f"- Spend: {f('kpi_spend').render()} ({f('kpi_spend_wow').render()} WoW)",
        f"- Signups: {f('kpi_signups').render()} ({f('kpi_signups_wow').render()} WoW)",
        f"- Paid signups: {f('kpi_paid_signups').render()} "
        f"({f('kpi_paid_signups_wow').render()} WoW)",
        f"- Revenue: {f('kpi_revenue').render()} ({f('kpi_revenue_wow').render()} WoW)",
        f"- Blended CAC: {f('kpi_cac').render()} ({f('kpi_cac_wow').render()} WoW)",
        f"- Fraud rate: {f('kpi_fraud_rate').render()} ({f('kpi_fraud_rate_wow').render()} WoW)",
        "",
        "## Channel performance, trailing week",
        "",
        f("channel_week").render(),
        "",
        "## Customer lifetime value",
        "",
        f("ltv_plans").render(),
        "",
        f("ltv_channels").render(),
        "",
        "## Outlook and monitoring",
        "",
        f"- Expected revenue over the next {f('forecast_horizon').render()} days: "
        f"{f('forecast_total').render()}",
        f"- Anomalous revenue days flagged: {f('anomaly_count').render()}",
        "",
        "## Provenance",
        "",
        "Every number above is a tracked figure; sources follow.",
        "",
    ]
    lines.extend(f"- `{fig.slug}` — {fig.source}" for fig in figures.all())
    return "\n".join(lines)


def build_weekly_review(con: duckdb.DuckDBPyConnection) -> GrowthReview:
    row = con.execute("SELECT max(date) FROM marts.mart_daily_channel").fetchone()
    if row is None or row[0] is None:
        raise ValueError("warehouse has no dated data in marts.mart_daily_channel")
    as_of: date = row[0] if isinstance(row[0], date) else row[0].date()
    week_start = as_of - timedelta(days=6)

    figures = FigureRegistry()
    _kpi_figures(con, figures, week_start, as_of)
    _channel_figure(con, figures, week_start)
    _ltv_figures(con, figures, as_of)
    _forecast_and_anomaly_figures(con, figures)
    return GrowthReview(as_of=as_of, markdown=_render(figures, as_of, week_start), figures=figures)
