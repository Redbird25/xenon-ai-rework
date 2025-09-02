# AI Ingest Service (MVP)

## Run locally
```bash
uvicorn app.main:app --reload --port 8001
```

## Env variables
- DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/xenon_ai_db
- GEMINI_API_KEY=...

## Test ingestion
```bash
curl -X POST http://localhost:8001/ingest/resources   -H "Content-Type: application/json"   -d '{
    "lesson_id": "c56a4180-65aa-42ec-a945-5fd21dec0538",
    "text": "This is a simple lesson about variables in Java.",
    "source_ref": "manual"
  }'
```
