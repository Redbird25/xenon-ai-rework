"""
Main FastAPI application with modern middleware and monitoring
"""
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import time
import uuid
from contextlib import asynccontextmanager
from sqlalchemy import text
from app.models import IngestJob, Document, SearchQuery

from app.config import settings
from app.core.logging import get_logger, setup_logging, request_id_var
from app.core.exceptions import AIIngestException
from app.routes import ingest_routes, search_routes
from app.db import engine
 

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    setup_logging()
    logger.info("AI Ingest Service starting up",
               environment=settings.environment.value,
               debug=settings.debug)
    # Ensure minimum schema exists (safe, idempotent)
    try:
        async with engine.begin() as conn:
            # Create missing tables for job tracking and search analytics
            await conn.run_sync(lambda sync_conn: IngestJob.__table__.create(sync_conn, checkfirst=True))
            await conn.run_sync(lambda sync_conn: Document.__table__.create(sync_conn, checkfirst=True))
            await conn.run_sync(lambda sync_conn: SearchQuery.__table__.create(sync_conn, checkfirst=True))
            # Note: Schema migrations for lesson_chunks are handled via migration files.
            # Avoid destructive or incompatible DDL at runtime.
    except Exception as e:
        logger.error("Schema ensure failed", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("AI Ingest Service shutting down")
    await engine.dispose()


app = FastAPI(
    title="AI Ingest Service",
    version="2.0.0",
    description="Modern document ingestion service with RAG capabilities",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests"""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_id_var.set(request_id)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    # Log request
    logger.info("request_received",
               method=request.method,
               path=request.url.path,
               client=request.client.host if request.client else None)
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Log response
        logger.info("request_completed",
                   method=request.method,
                   path=request.url.path,
                   status_code=response.status_code,
                   duration_seconds=duration)
        
        response.headers["X-Process-Time"] = str(duration)
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("request_failed",
                    method=request.method,
                    path=request.url.path,
                    error=str(e),
                    duration_seconds=duration)
        raise


@app.exception_handler(AIIngestException)
async def handle_ai_ingest_exception(request: Request, exc: AIIngestException):
    """Handle custom exceptions"""
    logger.error("Application error",
                error=exc.message,
                details=exc.details,
                path=request.url.path)
    
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "details": exc.details,
            "request_id": request_id_var.get()
        }
    )


@app.exception_handler(Exception)
async def handle_generic_exception(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error("Unexpected error",
                error=str(exc),
                error_type=type(exc).__name__,
                path=request.url.path)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An unexpected error occurred",
            "request_id": request_id_var.get()
        }
    )


# Include routers
app.include_router(ingest_routes.router)
app.include_router(search_routes.router)


@app.get("/health")
async def health():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "ai-ingest",
        "version": "2.0.0",
        "environment": settings.environment.value
    }


@app.get("/health/ready")
async def readiness():
    """Readiness check with database connectivity"""
    try:
        # Check database
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        
        return {
            "status": "ready",
            "checks": {
                "database": "ok"
            }
        }
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "not ready",
                "checks": {
                    "database": "failed"
                },
                "error": str(e)
            }
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Ingest Service",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }
