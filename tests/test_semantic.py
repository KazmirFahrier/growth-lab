from __future__ import annotations

import duckdb
import pytest

from growth_lab.warehouse.semantic import (
    METRICS,
    MetricFilters,
    compute_metrics,
    metric_query,
    parameterized_metric_query,
)


def test_unknown_metric_raises() -> None:
    with pytest.raises(KeyError):
        metric_query(["ctr", "made_up_metric"])


def test_unknown_or_duplicate_dimension_raises() -> None:
    with pytest.raises(ValueError, match="unknown dimension"):
        metric_query(["revenue"], by=["channel; SELECT 1"])
    with pytest.raises(ValueError, match="duplicates"):
        metric_query(["revenue"], by=["channel", "channel"])


def test_external_filters_are_parameterized() -> None:
    payload = "display' OR 1=1 --"
    sql, parameters = parameterized_metric_query(
        ["revenue"], by=["channel"], filters=MetricFilters(channel=payload)
    )
    assert payload not in sql
    assert "channel = ?" in sql
    assert parameters == (payload,)


def test_filter_dates_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="start_date"):
        MetricFilters.from_mapping({"start_date": "2026-02-02", "end_date": "2026-01-01"})


def test_every_metric_computes(warehouse: duckdb.DuckDBPyConnection) -> None:
    frame = compute_metrics(warehouse, sorted(METRICS))
    assert len(frame) == 1
    assert not frame.isna().any().any()


def test_ctr_is_ratio_of_sums(warehouse: duckdb.DuckDBPyConnection) -> None:
    """The semantic CTR must equal sum(clicks)/sum(impressions) computed from
    raw — not an average of daily ratios."""
    semantic = compute_metrics(warehouse, ["ctr"]).iloc[0, 0]
    row = warehouse.execute(
        "SELECT CAST(SUM(clicks) AS DOUBLE) / SUM(impressions) FROM raw.ad_spend_daily"
    ).fetchone()
    assert row is not None
    assert semantic == pytest.approx(row[0], rel=1e-12)


def test_mart_agrees_with_raw_signups(warehouse: duckdb.DuckDBPyConnection) -> None:
    mart_total = compute_metrics(warehouse, ["signups"]).iloc[0, 0]
    row = warehouse.execute("SELECT count(*) FROM raw.signups").fetchone()
    assert row is not None
    assert int(mart_total) == int(row[0])


def test_mart_revenue_agrees_with_raw(warehouse: duckdb.DuckDBPyConnection) -> None:
    mart_rev = compute_metrics(warehouse, ["revenue"]).iloc[0, 0]
    row = warehouse.execute("SELECT SUM(amount) FROM raw.transactions").fetchone()
    assert row is not None
    assert mart_rev == pytest.approx(row[0], rel=1e-9)


def test_organic_has_no_spend(warehouse: duckdb.DuckDBPyConnection) -> None:
    frame = compute_metrics(warehouse, ["spend", "signups"], by=["channel"])
    organic = frame[frame["channel"] == "organic"]
    assert len(organic) == 1
    assert organic["spend"].iloc[0] == 0.0
    assert organic["signups"].iloc[0] > 0


def test_metrics_by_channel_covers_all_channels(warehouse: duckdb.DuckDBPyConnection) -> None:
    frame = compute_metrics(warehouse, ["cac"], by=["channel"])
    assert set(frame["channel"]) == {"search", "social", "display", "video", "organic"}
