# syntax=docker/dockerfile:1.7

########## Builder stage ##########
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Системные пакеты для сборки wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Копируем зависимости и устанавливаем
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip && \
    pip install -r requirements.txt

########## Runtime stage ##########
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PORT=8001 \
    UVICORN_WORKERS=2

WORKDIR /app

# Только рантайм зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Копируем установленный питон-стек из билдера
COPY --from=builder /usr/local /usr/local

# Копируем исходники
COPY . .

# Создаем пользователя без root-прав
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

# Значения по умолчанию (можно переопределить в docker-compose/k8s)
ENV CORE_CALLBACK_URL=http://course-service:8082/api/course/ingest/callback

EXPOSE 8001

# Healthcheck — лучше завести лёгкий /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}"]
