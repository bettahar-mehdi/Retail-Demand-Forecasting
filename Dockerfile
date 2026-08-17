FROM python:3.10-slim AS base

# System deps for PySpark / Hadoop native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless procps && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYSPARK_PYTHON=python

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ src/
COPY conf/ conf/
COPY pyproject.toml .

EXPOSE 8000

# Default: start the FastAPI server
CMD ["uvicorn", "retail_demand_forecasting.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
