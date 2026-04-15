# syntax=docker/dockerfile:1.7
# Multi-stage production build for the RAG platform API

# ---- base ---------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---- deps ---------------------------------------------------------------
FROM base AS deps

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- builder ------------------------------------------------------------
FROM deps AS builder

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-deps -e .

# ---- runtime ------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Non-root user
RUN groupadd --gid 1001 rag && \
    useradd --uid 1001 --gid rag --shell /bin/bash --create-home rag

# Copy installed packages and app code
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY --from=builder /app/src ./src

USER rag

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

EXPOSE 8000

CMD ["uvicorn", "src.rag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
