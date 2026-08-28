# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# Stage 1: builder — install deps in virtualenv
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Build deps for lightgbm / xgboost if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src/ src/
COPY conf/ conf/

# Create venv and install — use pyproject for full pinning
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install -r requirements.txt && \
    /opt/venv/bin/pip install -e . --no-deps

# ---------------------------------------------------------------------------
# Stage 2: runtime — minimal, non-root, both FastAPI + Dash
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYSPARK_PYTHON=python \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH"

# Runtime deps: JRE for PySpark, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 -s /bin/bash appuser

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy app code (conf, src, entrypoints)
COPY --chown=appuser:appuser conf/ conf/
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser start_all.py ./
COPY --chown=appuser:appuser models/ models/ 2>/dev/null || mkdir -p models

# Create data dirs (DVC will populate via dvc pull at runtime if needed)
RUN mkdir -p data/01_raw data/02_intermediate data/03_features data/06_metrics data/processed \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8050

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run both API (8000) and Dash (8050) via start_all.py
# Override with: docker run ... uvicorn retail_demand_forecasting.api.app:app --host 0.0.0.0 --port 8000
CMD ["python", "start_all.py"]
