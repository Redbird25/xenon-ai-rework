# --- Base image ---
FROM python:3.13-slim

# --- Set workdir ---
WORKDIR /app

# --- Install system deps (for pgvector, psycopg2, etc.) ---
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Copy dependencies ---
COPY requirements.txt .

# --- Install python deps ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project files ---
COPY . .

# --- Set env vars (можно переопределять через docker-compose/k8s) ---
ENV PYTHONUNBUFFERED=1
ENV CORE_CALLBACK_URL=http://core-service:8080/api/ingest/callback

# --- Expose FastAPI port ---
EXPOSE 8000

# --- Run server ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
