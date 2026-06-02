# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------
# Agentic RAG Compliance Assistant - application image
# Uses `uv` for fast, reproducible installs from uv.lock.
# ---------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# System deps for PDF parsing (pdfplumber needs a few shared libs)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single static binary)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install dependencies first (layer cache friendly): copy only manifests
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy source
COPY app ./app

# Install the project itself (skip dev extras for slimmer image)
RUN uv sync --frozen --no-dev

EXPOSE 8501

# Streamlit needs to bind to 0.0.0.0 inside containers
CMD ["streamlit", "run", "app/ui/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
