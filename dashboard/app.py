"""Meridian growth dashboard.

Run:
    pip install -e ".[dashboard]"
    python -m growth_lab build            # creates data/growth_lab.duckdb
    streamlit run dashboard/app.py

Set GROWTH_LAB_DB to point at a different warehouse file.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from growth_lab.experiments import ArmCounts, ExperimentDesign, evaluate, to_markdown
from growth_lab.forecasting import HoltWinters
from growth_lab.marketing import ChannelResponse, fit_mmm, optimal_allocation
from growth_lab.risk import mad_residual_detector
from growth_lab.simulator.scenarios import mmm_market
from growth_lab.warehouse.semantic import compute_metrics

DB_PATH = Path(os.environ.get("GROWTH_LAB_DB", "data/growth_lab.duckdb"))

st.set_page_config(page_title="Meridian growth lab", layout="wide")
st.title("Meridian growth lab")

if not DB_PATH.exists():
    st.error(f"No warehouse at `{DB_PATH}`. Run `python -m growth_lab build` first.")
    st.stop()


@st.cache_resource
def connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


@st.cache_data
def revenue_series() -> pd.DataFrame:
    return (
        connection()
        .execute(
            "SELECT date, SUM(revenue) AS revenue FROM marts.mart_daily_channel "
            "GROUP BY date ORDER BY date"
        )
        .df()
    )


overview, channels_tab, mmm_tab, forecast_tab, experiment_tab = st.tabs(
    ["Overview", "Channels", "MMM & budget", "Forecast & anomalies", "Experiment readout"]
)

with overview:
    kpis = compute_metrics(
        connection(), ["spend", "signups", "paid_signups", "revenue", "cac", "fraud_rate"]
    )
    cols = st.columns(6)
    labels = ["Spend", "Signups", "Paid", "Revenue", "CAC", "Fraud rate"]
    formats = ["${:,.0f}", "{:,.0f}", "{:,.0f}", "${:,.0f}", "${:,.2f}", "{:.2%}"]
    for col, label, fmt, key in zip(cols, labels, formats, kpis.columns, strict=True):
        col.metric(label, fmt.format(float(kpis[key].iloc[0])))
    st.caption("All metrics are ratio-of-sums from the semantic layer.")

with channels_tab:
    frame = compute_metrics(
        connection(),
        ["spend", "signups", "paid_signups", "cac", "revenue"],
        by=["channel"],
    )
    st.dataframe(frame, use_container_width=True)
    st.bar_chart(frame.set_index("channel")["revenue"])
    st.caption(
        "Display's CAC looks unbeatable — Phase 2/3 show that is the "
        "intent-harvesting illusion, not incrementality."
    )

with mmm_tab:
    st.write(
        "MMM fit on the media-market scenario (the 180-day warehouse has "
        "constant spend, which cannot identify response curves — a lesson in "
        "itself)."
    )

    @st.cache_resource
    def fitted_mmm():  # type: ignore[no-untyped-def]
        scenario = mmm_market()
        fit = fit_mmm(scenario.spend, scenario.revenue, scenario.day_of_week, scenario.channels)
        return scenario, fit

    scenario, fit = fitted_mmm()
    st.write(f"R² {fit.r_squared:.3f} — decay {np.round(fit.decay, 2).tolist()}")
    curves = pd.DataFrame(
        {
            name: fit.beta[c]
            * (np.linspace(0, 3000, 100) / (1 - fit.decay[c]))
            / (np.linspace(0, 3000, 100) / (1 - fit.decay[c]) + fit.half_sat[c])
            for c, name in enumerate(scenario.channels)
        },
        index=np.linspace(0, 3000, 100),
    )
    st.line_chart(curves)

    budget = st.slider("Total daily budget ($)", 500, 5000, 1800, step=100)
    responses = tuple(
        ChannelResponse(name, float(fit.beta[c]), float(fit.decay[c]), float(fit.half_sat[c]))
        for c, name in enumerate(scenario.channels)
    )
    allocation = optimal_allocation(responses, float(budget))
    alloc_frame = pd.DataFrame(
        {"channel": allocation.channels, "daily_spend": allocation.daily_spend}
    ).set_index("channel")
    st.bar_chart(alloc_frame)
    st.write(f"Expected steady-state daily revenue: ${allocation.expected_daily_revenue:,.0f}")

with forecast_tab:
    series = revenue_series()
    revenue = series["revenue"].to_numpy(dtype=np.float64)
    model = HoltWinters()
    model.fit(revenue)
    horizon = 14
    forecast = model.predict(horizon)
    history = pd.DataFrame({"revenue": revenue}, index=pd.to_datetime(series["date"]))
    future_idx = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon)
    combined = pd.concat([history, pd.DataFrame({"forecast": forecast}, index=future_idx)])
    st.line_chart(combined)

    detector = mad_residual_detector(revenue)
    if len(detector.flagged_days):
        st.dataframe(series.iloc[detector.flagged_days])
    else:
        st.success("No anomalous revenue days at the current threshold.")

with experiment_tab:
    st.write("Interactive readout: paste your arm counts, get a launch decision.")
    c1, c2 = st.columns(2)
    n_control = c1.number_input("Control n", 100, 10_000_000, 20_000)
    x_control = c1.number_input("Control conversions", 0, 10_000_000, 1_000)
    n_treatment = c2.number_input("Treatment n", 100, 10_000_000, 20_000)
    x_treatment = c2.number_input("Treatment conversions", 0, 10_000_000, 1_100)
    mde = st.slider("Design MDE (relative)", 0.01, 0.5, 0.10)

    design = ExperimentDesign(
        name="dashboard",
        primary_metric="conversion",
        baseline_rate=max(float(x_control) / float(n_control), 1e-6),
        mde_relative=float(mde),
    )
    readout = evaluate(
        design,
        ArmCounts(int(n_control), int(x_control), {}),
        ArmCounts(int(n_treatment), int(x_treatment), {}),
    )
    st.markdown(to_markdown(readout))
