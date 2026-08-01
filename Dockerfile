# ── Build stage: install dependencies ───────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir uvicorn[standard]

# ── Runtime stage: minimal production image ──────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ src/
COPY dbt/ dbt/
COPY truth.yaml .
COPY README.md .

# Copy model artifacts (fallback: create empty dir so startup doesn't crash)
COPY models/ models/
RUN mkdir -p /app/models

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')" || exit 1

CMD ["uvicorn", "growth_lab.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
