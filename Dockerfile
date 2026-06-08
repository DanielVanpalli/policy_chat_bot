# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

WORKDIR /app

# System deps for spacy (presidio) and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# ── Layer 1: dependencies (invalidates only when pyproject/uv.lock change) ──
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-hashes --no-emit-project > /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt

# ── Layer 2: spacy model (heavy; cache this separately from source) ──
RUN python -m spacy download en_core_web_lg

# ── Layer 3: project source (changes often) ──
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
