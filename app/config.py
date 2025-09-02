import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:admin@75.119.145.146:5433/xenon_ai_db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBAkY3izlayY0oQtbEk46_tz7Bss0fQLd8")

EMBEDDING_DIM = 1536

CORE_CALLBACK_URL = os.getenv("CORE_CALLBACK_URL", "http://core-service:8080/api/ingest/callback")
