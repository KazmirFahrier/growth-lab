"""Service health monitoring: latency, throughput, and data freshness.

Tracks operational metrics for the prediction service. Designed to be called
from the /metrics Prometheus endpoint or from a scheduled monitoring job.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
from datetime import time as datetime_time
from pathlib import Path

import duckdb


@dataclass
class LatencyReport:
    """Prediction service latency snapshot."""

    checked_at: datetime
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_requests: int
    error_rate: float


@dataclass
class ThroughputReport:
    """Prediction service throughput snapshot."""

    checked_at: datetime
    requests_per_second: float
    total_requests: int
    window_seconds: float


@dataclass
class DataFreshnessReport:
    """Warehouse data freshness check."""

    checked_at: datetime
    latest_txn_date: str
    latest_signup_date: str
    freshness_hours: float
    is_fresh: bool
    warning: str = ""


class HealthMonitor:
    """Collect and report service health metrics."""

    def __init__(self, latency_window: int = 10_000) -> None:
        if latency_window < 100:
            raise ValueError("latency_window must be at least one hundred")
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        self._latencies: deque[float] = deque(maxlen=latency_window)

    def record(self, latency_s: float, is_error: bool = False) -> None:
        """Record one prediction request."""
        if latency_s < 0:
            raise ValueError("latency must not be negative")
        self._request_count += 1
        self._latencies.append(latency_s)
        if is_error:
            self._error_count += 1

    def latency_report(self) -> LatencyReport:
        now = datetime.now(timezone.utc)
        if not self._latencies:
            return LatencyReport(
                checked_at=now,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                n_requests=0,
                error_rate=0.0,
            )
        arr = sorted(self._latencies)
        n = len(arr)
        return LatencyReport(
            checked_at=now,
            p50_ms=round(arr[int(n * 0.50)] * 1000, 2),
            p95_ms=round(arr[min(int(n * 0.95), n - 1)] * 1000, 2),
            p99_ms=round(arr[min(int(n * 0.99), n - 1)] * 1000, 2),
            n_requests=self._request_count,
            error_rate=round(self._error_count / max(self._request_count, 1), 4),
        )

    def throughput_report(self) -> ThroughputReport:
        now = datetime.now(timezone.utc)
        elapsed = time.time() - self._start_time
        return ThroughputReport(
            checked_at=now,
            requests_per_second=round(self._request_count / max(elapsed, 0.001), 2),
            total_requests=self._request_count,
            window_seconds=round(elapsed, 1),
        )

    def data_freshness(
        self,
        db_path: Path,
        max_age_hours: float = 48.0,
        reference_time: datetime | None = None,
    ) -> DataFreshnessReport:
        """Check how fresh the warehouse data is."""
        if max_age_hours <= 0:
            raise ValueError("max_age_hours must be positive")
        now = reference_time or datetime.now(timezone.utc)
        con: duckdb.DuckDBPyConnection | None = None
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            max_txn = con.execute("SELECT MAX(txn_date) FROM raw.transactions").fetchone()
            max_signup = con.execute("SELECT MAX(signup_date) FROM raw.signups").fetchone()
            if not max_txn or not isinstance(max_txn[0], (date, datetime)):
                raise ValueError("warehouse has no transaction timestamp")
            txn_value = max_txn[0]
            latest_datetime = (
                txn_value
                if isinstance(txn_value, datetime)
                else datetime.combine(txn_value, datetime_time.min, tzinfo=timezone.utc)
            )
            if latest_datetime.tzinfo is None:
                latest_datetime = latest_datetime.replace(tzinfo=timezone.utc)
            freshness_hours = max(0.0, (now - latest_datetime).total_seconds() / 3600.0)
            latest_txn = str(txn_value)
            latest_signup = str(max_signup[0]) if max_signup and max_signup[0] else "unknown"
            is_fresh = freshness_hours <= max_age_hours
            warning = "" if is_fresh else f"warehouse is {freshness_hours:.1f} hours old"
        except (OSError, ValueError, duckdb.Error) as error:
            latest_txn = "error"
            latest_signup = "error"
            freshness_hours = float("inf")
            is_fresh = False
            warning = str(error)
        finally:
            if con is not None:
                con.close()

        return DataFreshnessReport(
            checked_at=now,
            latest_txn_date=latest_txn,
            latest_signup_date=latest_signup,
            freshness_hours=freshness_hours,
            is_fresh=is_fresh,
            warning=warning,
        )


def latency_report(monitor: HealthMonitor) -> LatencyReport:
    return monitor.latency_report()


def throughput_report(monitor: HealthMonitor) -> ThroughputReport:
    return monitor.throughput_report()
