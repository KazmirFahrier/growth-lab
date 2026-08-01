"""Bounded HTTP telemetry with Prometheus text export."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class RequestMetrics:
    """Thread safe counters for the small, fixed API route set."""

    _counts: dict[tuple[str, str, int], int] = field(default_factory=lambda: defaultdict(int))
    _latency_seconds: dict[tuple[str, str], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    _lock: Lock = field(default_factory=Lock, repr=False)

    def observe(self, method: str, route: str, status: int, elapsed_seconds: float) -> None:
        with self._lock:
            self._counts[(method, route, status)] += 1
            self._latency_seconds[(method, route)] += elapsed_seconds

    def render(self) -> str:
        lines = [
            "# HELP growth_lab_http_requests_total Total HTTP responses.",
            "# TYPE growth_lab_http_requests_total counter",
        ]
        with self._lock:
            counts = sorted(self._counts.items())
            latency = sorted(self._latency_seconds.items())
        for (method, route, status), value in counts:
            labels = f'method="{method}",route="{route}",status="{status}"'
            lines.append(f"growth_lab_http_requests_total{{{labels}}} {value}")
        lines.extend(
            [
                "# HELP growth_lab_http_request_duration_seconds_sum Total request time.",
                "# TYPE growth_lab_http_request_duration_seconds_sum counter",
            ]
        )
        for (method, route), latency_value in latency:
            labels = f'method="{method}",route="{route}"'
            lines.append(
                f"growth_lab_http_request_duration_seconds_sum{{{labels}}} {latency_value:.6f}"
            )
        return "\n".join(lines) + "\n"
