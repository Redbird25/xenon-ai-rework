########## Builder stage ##########
FROM python:3.13-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

########## Runtime stage ##########
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=1 \
    PORT=8001 \
    UVICORN_WORKERS=2

WORKDIR /app

# Runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libxml2 \
    libxslt1.1 \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy source
COPY . .

# Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

# Defaults (override via env)
ENV CORE_CALLBACK_URL=http://course-service:8082/api/course/ingest/callback

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/openapi.json > /dev/null || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}"]
