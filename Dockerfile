FROM python:3.14.6-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
RUN python -m venv /opt/growth-lab
ENV PATH="/opt/growth-lab/bin:${PATH}" \
    GROWTH_LAB_TRUTH_PATH=/build/truth.yaml \
    GROWTH_LAB_DBT_DIR=/build/dbt

COPY pyproject.toml constraints.txt README.md truth.yaml ./
COPY src ./src
COPY dbt ./dbt

RUN pip install ".[service]" -c constraints.txt
RUN python -m growth_lab build --db /build/data/growth_lab.duckdb \
    && python -m growth_lab export-mmm --out /build/models/mmm.json

FROM builder AS churn-builder

ENV GROWTH_LAB_MODEL_DIR=/build/models \
    MLFLOW_TRACKING_URI=sqlite:////build/mlflow.db \
    GIT_PYTHON_REFRESH=quiet
RUN pip install ".[ml]" -c constraints.txt
RUN growth-lab-train \
    --db /build/data/growth_lab.duckdb \
    --cutoff-days 60 \
    --horizon-days 30 \
    --no-register

FROM python:3.14.6-slim AS runtime-base

ENV PATH="/opt/growth-lab/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GROWTH_LAB_ENV=production \
    GROWTH_LAB_MAX_REQUEST_BYTES=1048576 \
    GROWTH_LAB_LOG_LEVEL=INFO

RUN groupadd --gid 10001 growthlab \
    && useradd --uid 10001 --gid growthlab --no-create-home --shell /usr/sbin/nologin growthlab
WORKDIR /app

FROM runtime-base AS analytics

ENV GROWTH_LAB_DB=/app/data/growth_lab.duckdb \
    GROWTH_LAB_MMM_PARAMS=/app/models/mmm.json
COPY --from=builder /opt/growth-lab /opt/growth-lab
COPY --from=builder --chown=growthlab:growthlab /build/data ./data
COPY --from=builder --chown=growthlab:growthlab /build/models/mmm.json ./models/mmm.json
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["uvicorn", "growth_lab.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--no-access-log"]

FROM runtime-base AS churn

ENV GROWTH_LAB_DB=/app/data/growth_lab.duckdb \
    GROWTH_LAB_CHURN_MODEL=/app/models/churn_model.joblib \
    GROWTH_LAB_FEATURE_NAMES=/app/models/feature_names.json
COPY --from=churn-builder /opt/growth-lab /opt/growth-lab
COPY --from=churn-builder --chown=growthlab:growthlab /build/data ./data
COPY --from=churn-builder --chown=growthlab:growthlab /build/models ./models
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"]
CMD ["uvicorn", "growth_lab.serve.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--no-access-log"]

FROM analytics AS production
