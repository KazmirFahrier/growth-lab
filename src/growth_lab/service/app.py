"""Authenticated FastAPI boundary for Growth Lab capabilities."""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

import duckdb
from fastapi import Depends, FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from growth_lab import __version__
from growth_lab.integrations import (
    BudgetPlannerTool,
    ForecastRevenueTool,
    QueryGrowthMetricsTool,
    ToolResult,
)
from growth_lab.service.config import Settings
from growth_lab.service.observability import RequestMetrics

REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log event."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in ("request_id", "method", "route", "status", "elapsed_ms"):
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        return json.dumps(payload, separators=(",", ":"))


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("growth_lab")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(level)


class _RequestTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """Enforce a body limit for declared and streamed request bodies."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                declared_too_large = int(raw_length) > self.max_bytes
            except ValueError:
                declared_too_large = True
            if declared_too_large:
                request_id = scope.get("state", {}).get("request_id", uuid.uuid4().hex)
                response = JSONResponse(
                    status_code=413,
                    content={"error": "REQUEST_TOO_LARGE", "request_id": request_id},
                )
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestTooLarge:
            request_id = scope.get("state", {}).get("request_id", uuid.uuid4().hex)
            response = JSONResponse(
                status_code=413,
                content={"error": "REQUEST_TOO_LARGE", "request_id": request_id},
            )
            await response(scope, receive, send)


class RuntimeMiddleware(BaseHTTPMiddleware):
    """Apply request limits, correlation IDs, security headers, and telemetry."""

    def __init__(self, app: Any, settings: Settings, metrics: RequestMetrics) -> None:
        super().__init__(app)
        self.settings = settings
        self.metrics = metrics
        self.logger = logging.getLogger("growth_lab.http")

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        supplied_id = request.headers.get("X-Request-ID", "")
        request_id = supplied_id if REQUEST_ID.fullmatch(supplied_id) else uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        route = getattr(request.scope.get("route"), "path", "unmatched")
        if route == "unmatched":
            route = "unmatched"
        self.metrics.observe(request.method, route, response.status_code, elapsed)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        self.logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "route": route,
                "status": response.status_code,
                "elapsed_ms": round(elapsed * 1000.0, 3),
            },
        )
        return response


class MetricFilterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str | None = Field(default=None, max_length=128)
    start_date: date | None = None
    end_date: date | None = None


class MetricsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(min_length=1, max_length=20)
    by: list[str] = Field(default_factory=list, max_length=2)
    filters: MetricFilterRequest | None = None


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon_days: int = Field(ge=1, le=56)


class BudgetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_daily_budget: float = Field(gt=0, le=1_000_000_000, allow_inf_nan=False)


def _status_for(result: ToolResult) -> int:
    if result.ok:
        return 200
    if result.error_code in {
        "INVALID_METRIC",
        "INVALID_DIMENSION",
        "INVALID_FILTER",
        "INVALID_HORIZON",
        "INVALID_BUDGET",
        "UNSAFE_FILTER",
    }:
        return 422
    if result.error_code in {
        "WAREHOUSE_UNAVAILABLE",
        "WAREHOUSE_EMPTY",
        "INSUFFICIENT_HISTORY",
        "NO_MMM_PARAMS",
        "INVALID_MMM_PARAMS",
    }:
        return 503
    return 500


def _tool_response(result: ToolResult, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=_status_for(result),
        content=jsonable_encoder(
            {
                "ok": result.ok,
                "content": result.content,
                "data": result.data,
                "error_code": result.error_code,
                "request_id": request_id,
            }
        ),
    )


def _readiness_checks(settings: Settings) -> dict[str, bool]:
    checks = {
        "warehouse": False,
        "mmm_params": BudgetPlannerTool(settings.mmm_params_path).run(total_daily_budget=1.0).ok,
    }
    if settings.db_path.is_file():
        try:
            con = duckdb.connect(str(settings.db_path), read_only=True)
            try:
                con.execute("SELECT 1 FROM marts.mart_daily_channel LIMIT 1").fetchone()
                checks["warehouse"] = True
            finally:
                con.close()
        except duckdb.Error:
            pass
    return checks


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime = settings or Settings.from_env()
    configure_logging(runtime.log_level)
    logger = logging.getLogger("growth_lab.service")
    metrics = RequestMetrics()
    app = FastAPI(
        title="Growth Lab API",
        version=__version__,
        docs_url=None if runtime.environment == "production" else "/docs",
        redoc_url=None,
    )
    app.state.settings = runtime
    app.state.metrics = metrics
    app.add_middleware(BodyLimitMiddleware, max_bytes=runtime.max_request_bytes)
    app.add_middleware(RuntimeMiddleware, settings=runtime, metrics=metrics)

    async def authorize(x_api_key: str | None = Header(default=None)) -> None:
        if runtime.api_key is None:
            return
        if x_api_key is None or not secrets.compare_digest(x_api_key, runtime.api_key):
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="invalid API key")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, error: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        logger.exception("unhandled request error", extra={"request_id": request_id})
        return JSONResponse(
            status_code=500,
            content={"error": "INTERNAL_ERROR", "request_id": request_id},
        )

    @app.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", include_in_schema=False)
    async def ready() -> JSONResponse:
        checks = await run_in_threadpool(_readiness_checks, runtime)
        status = 200 if all(checks.values()) else 503
        return JSONResponse(status_code=status, content={"ready": status == 200, "checks": checks})

    @app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    async def operational_metrics(_: None = Depends(authorize)) -> str:
        return metrics.render()

    @app.post("/v1/metrics")
    async def business_metrics(
        body: MetricsRequest,
        request: Request,
        _: None = Depends(authorize),
    ) -> JSONResponse:
        tool = QueryGrowthMetricsTool(runtime.db_path)
        filters = body.filters.model_dump(mode="json", exclude_none=True) if body.filters else None
        result = await run_in_threadpool(
            tool.run,
            metrics=body.metrics,
            by=body.by,
            filters=filters,
        )
        return _tool_response(result, request.state.request_id)

    @app.post("/v1/forecast")
    async def forecast(
        body: ForecastRequest,
        request: Request,
        _: None = Depends(authorize),
    ) -> JSONResponse:
        result = await run_in_threadpool(
            ForecastRevenueTool(runtime.db_path).run,
            horizon_days=body.horizon_days,
        )
        return _tool_response(result, request.state.request_id)

    @app.post("/v1/budget")
    async def budget(
        body: BudgetRequest,
        request: Request,
        _: None = Depends(authorize),
    ) -> JSONResponse:
        result = await run_in_threadpool(
            BudgetPlannerTool(runtime.mmm_params_path).run,
            total_daily_budget=body.total_daily_budget,
        )
        return _tool_response(result, request.state.request_id)

    return app
