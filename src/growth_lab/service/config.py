"""Runtime configuration loaded from explicit environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Validated service settings.

    Production mode refuses to start without a strong API key. Development
    and test modes may run without authentication for local workflows.
    """

    environment: str = "development"
    db_path: Path = Path("data/growth_lab.duckdb")
    mmm_params_path: Path = Path("models/mmm.json")
    api_key: str | None = field(default=None, repr=False)
    max_request_bytes: int = 1_048_576
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.environment not in {"development", "test", "production"}:
            raise ValueError("environment must be development, test, or production")
        if self.environment == "production" and (self.api_key is None or len(self.api_key) < 32):
            raise ValueError("production requires GROWTH_LAB_API_KEY with at least 32 characters")
        if self.max_request_bytes < 1024:
            raise ValueError("max_request_bytes must be at least 1024")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("unsupported log level")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        values = os.environ if environ is None else environ
        raw_limit = values.get("GROWTH_LAB_MAX_REQUEST_BYTES", "1048576")
        try:
            limit = int(raw_limit)
        except ValueError as error:
            raise ValueError("GROWTH_LAB_MAX_REQUEST_BYTES must be an integer") from error
        return cls(
            environment=values.get("GROWTH_LAB_ENV", "development").lower(),
            db_path=Path(values.get("GROWTH_LAB_DB", "data/growth_lab.duckdb")),
            mmm_params_path=Path(values.get("GROWTH_LAB_MMM_PARAMS", "models/mmm.json")),
            api_key=values.get("GROWTH_LAB_API_KEY"),
            max_request_bytes=limit,
            log_level=values.get("GROWTH_LAB_LOG_LEVEL", "INFO").upper(),
        )
