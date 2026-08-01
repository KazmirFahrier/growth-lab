"""Service health monitoring: latency, throughput, and data freshness.

Tracks operational metrics for the prediction service. Designed to be called
from the /metrics Prometheus endpoint or from a scheduled monitoring job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

    def __init__(self) -> None:
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        self._latencies: list[float] = []

    def record(self, latency_s: float, is_error: bool = False) -> None:
        """Record one prediction request."""
        self._request_count += 1
        self._latencies.append(latency_s)
        if is_error:
            self._error_count += 1

    def latency_report(self) -> LatencyReport:
        now = datetime.now(timezone.utc)
        if not self._latencies:
            return LatencyReport(
                checked_at=now,
                p50_ms=0, p95_ms=0, p99_ms=0, n_requests=0, error_rate=0.0,
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

    def data_freshness(self, db_path: Path) -> DataFreshnessReport:
        """Check how fresh the warehouse data is."""
        now = datetime.now(timezone.utc)
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            max_txn = con.execute("SELECT MAX(txn_date) FROM raw.transactions").fetchone()
            max_signup = con.execute("SELECT MAX(signup_date) FROM raw.signups").fetchone()
            con.close()

            latest_txn = str(max_txn[0]) if max_txn and max_txn[0] else "unknown"
            latest_signup = str(max_signup[0]) if max_signup and max_signup[0] else "unknown"

            # For simulated data, a "fresh" check is approximate
            is_fresh = True
            warning = ""
        except Exception as e:
            latest_txn = "error"
            latest_signup = "error"
            is_fresh = False
            warning = str(e)

        return DataFreshnessReport(
            checked_at=now,
            latest_txn_date=latest_txn,
            latest_signup_date=latest_signup,
            freshness_hours=0.0,
            is_fresh=is_fresh,
            warning=warning,
        )


def latency_report(monitor: HealthMonitor) -> LatencyReport:
    return monitor.latency_report()


def throughput_report(monitor: HealthMonitor) -> ThroughputReport:
    return monitor.throughput_report()
