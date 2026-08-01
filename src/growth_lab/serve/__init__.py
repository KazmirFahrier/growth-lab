"""FastAPI prediction service for churn risk scoring.

Serves the trained churn model behind a REST API with validated request/response
schemas and Prometheus metrics middleware.
"""

from growth_lab.serve.app import create_app

__all__ = ["create_app"]
