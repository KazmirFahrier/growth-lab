"""Production HTTP service for governed Growth Lab models and metrics."""

from growth_lab.service.app import create_app
from growth_lab.service.config import Settings

__all__ = ["Settings", "create_app"]
