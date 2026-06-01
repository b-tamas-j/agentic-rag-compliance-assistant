# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------
# Agentic RAG Compliance Assistant - application image
# ---------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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

# Install Python deps first to leverage Docker layer cache
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy source
COPY app ./app
COPY ui ./ui
COPY eval ./eval

EXPOSE 8501

# Streamlit needs to bind to 0.0.0.0 inside containers
CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
