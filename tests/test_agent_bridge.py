"""Phase 6 gate (bridge): growth-lab tools satisfy the agent contract and
their answers are groundable — every number a tool states in its content is
licensed by its own data, exactly what campaign-copilot's grounding checker
demands.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pytest

from growth_lab.integrations import (
    BudgetPlannerTool,
    ForecastRevenueTool,
    LtvSummaryTool,
    QueryGrowthMetricsTool,
    Tool,
    ToolResult,
    default_toolkit,
)
from growth_lab.marketing import fit_mmm
from growth_lab.simulator.scenarios import mmm_market

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _content_is_grounded(result: ToolResult) -> bool:
    """Every numeric token in content must match a licensed fact."""
    facts = result.numeric_facts()
    for token in NUMBER.findall(result.content):
        value = abs(float(token.replace(",", "")))
        if not any(abs(value - abs(fact)) <= max(0.01, 0.005 * abs(fact)) for fact in facts):
            return False
    return True


@pytest.fixture(scope="module")
def db_path(warehouse: duckdb.DuckDBPyConnection, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    # the warehouse fixture's underlying file: rebuild a tiny pointer by
    # querying its path is not exposed, so land a fresh copy for the tools
    path = tmp_path_factory.mktemp("bridge") / "wh.duckdb"
    con = duckdb.connect(str(path))
    try:
        src = warehouse
        for schema, table in (
            ("marts", "mart_daily_channel"),
            ("marts", "dim_users"),
            ("marts", "fct_transactions"),
        ):
            frame = src.execute(f"SELECT * FROM {schema}.{table}").df()
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            con.register("_t", frame)
            con.execute(f"CREATE TABLE {schema}.{table} AS SELECT * FROM _t")
            con.unregister("_t")
    finally:
        con.close()
    return path


@pytest.fixture(scope="module")
def mmm_params(tmp_path_factory: pytest.TempPathFactory) -> Path:
    scenario = mmm_market()
    fit = fit_mmm(scenario.spend, scenario.revenue, scenario.day_of_week, scenario.channels)
    path = tmp_path_factory.mktemp("models") / "mmm.json"
    path.write_text(
        json.dumps(
            {
                "source": "test fit",
                "channels": [
                    {
                        "name": name,
                        "beta": float(fit.beta[c]),
                        "decay": float(fit.decay[c]),
                        "half_sat": float(fit.half_sat[c]),
                        "current_daily_spend": float(np.mean(scenario.spend[:, c])),
                    }
                    for c, name in enumerate(scenario.channels)
                ],
            }
        )
    )
    return path


def test_all_tools_satisfy_the_contract(db_path: Path, mmm_params: Path) -> None:
    for tool in default_toolkit(db_path, mmm_params):
        assert isinstance(tool, Tool)
        spec = tool.spec
        assert spec.name and spec.description
        assert spec.as_anthropic()["input_schema"] == spec.input_schema
        assert spec.as_openai()["function"]["name"] == spec.name


def test_metrics_tool_answers_and_grounds(db_path: Path) -> None:
    tool = QueryGrowthMetricsTool(db_path)
    result = tool.run(metrics=["cac", "revenue"], by=["channel"])
    assert result.ok
    assert result.data["rows"]
    assert "SELECT" in result.data["sql"]
    assert _content_is_grounded(result)


def test_metrics_tool_failures_are_results(db_path: Path) -> None:
    tool = QueryGrowthMetricsTool(db_path)
    bad_metric = tool.run(metrics=["made_up"])
    assert not bad_metric.ok and bad_metric.error_code == "INVALID_METRIC"
    injection = tool.run(metrics=["cac"], where="1=1; DROP TABLE x")
    assert not injection.ok and injection.error_code == "UNSAFE_FILTER"
    bad_dimension = tool.run(metrics=["cac"], by=["channel, sim_hidden.users_latent"])
    assert not bad_dimension.ok and bad_dimension.error_code == "INVALID_DIMENSION"
    missing = QueryGrowthMetricsTool(Path("nowhere.duckdb")).run(metrics=["cac"])
    assert not missing.ok and missing.error_code == "WAREHOUSE_UNAVAILABLE"


def test_metrics_tool_binds_external_filters(db_path: Path) -> None:
    tool = QueryGrowthMetricsTool(db_path)
    attempted_injection = "display' OR 1=1 --"
    result = tool.run(
        metrics=["revenue"],
        by=["channel"],
        filters={"channel": attempted_injection},
    )
    assert result.ok
    assert result.data["rows"] == []
    assert attempted_injection not in result.data["sql"]
    assert result.data["parameters"] == [attempted_injection]


def test_ltv_and_forecast_tools_work(db_path: Path) -> None:
    ltv = LtvSummaryTool(db_path).run()
    assert ltv.ok and len(ltv.data["plans"]) == 2

    forecast = ForecastRevenueTool(db_path).run(horizon_days=14)
    assert forecast.ok
    assert forecast.data["total"] == pytest.approx(sum(forecast.data["daily"]), rel=1e-6)
    bad = ForecastRevenueTool(db_path).run(horizon_days=999)
    assert not bad.ok and bad.error_code == "INVALID_HORIZON"


def test_budget_shift_demo_is_grounded(mmm_params: Path) -> None:
    """The flagship bridge demo: 'should we shift budget?' — answered from
    fitted response curves, with every stated number licensed by data."""
    tool = BudgetPlannerTool(mmm_params)
    params = json.loads(mmm_params.read_text())
    current_total = sum(c["current_daily_spend"] for c in params["channels"])

    result = tool.run(total_daily_budget=current_total)
    assert result.ok
    assert result.data["expected_daily_uplift"] >= 0  # optimum can't be worse
    assert len(result.data["shifts"]) == 4
    assert _content_is_grounded(result)


def test_budget_tool_without_params_gives_actionable_failure() -> None:
    result = BudgetPlannerTool(Path("missing/mmm.json")).run(total_daily_budget=1000)
    assert not result.ok
    assert result.error_code == "NO_MMM_PARAMS"
    assert "export-mmm" in result.content  # the repair instruction


def test_reference_results_license_no_numbers() -> None:
    result = ToolResult.reference("ROAS was 4.2x last quarter", doc="memo.md")
    assert result.numeric_facts() == []
