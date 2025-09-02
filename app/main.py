from fastapi import FastAPI
from app.routes import ingest_routes

app = FastAPI(title="AI Ingest Service")

app.include_router(ingest_routes.router)

@app.get("/health")
async def health():
    return {"status": "ok"}
