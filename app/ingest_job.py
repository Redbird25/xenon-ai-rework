"""
Ingestion job processing with modern architecture
"""
import httpx
import uuid
import time
from datetime import datetime
from typing import Dict, Any

from app.config import settings
from app.core.logging import get_logger, metrics_logger, job_id_var
from app.core.retry import http_retry, RetryWithBackoff
from app.ingest import get_ingest_service
from app.utils.resources import fetch_resources
from app.llm import generate_course_route
from app.db import async_session
from app.models import IngestJob

logger = get_logger(__name__)

# Retry decorator for callback
callback_retry = RetryWithBackoff(max_attempts=3, initial_delay=1.0)


@callback_retry
async def send_callback(result: Dict[str, Any]):
    """Send callback to core service with retry"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(settings.core_callback_url, json=result)
        response.raise_for_status()
        logger.info("Callback sent successfully", 
                   job_id=result.get("job_id"),
                   status=result.get("status"))


async def run_ingest_job(req, job_id: str):
    """
    Run document ingestion job with progress tracking
    """
    start_time = time.time()
    job_id_var.set(job_id)
    
    # Create job record
    async with async_session() as session:
        job = IngestJob(
            id=job_id,
            course_id=req.course_id,
            status="processing",
            job_type="batch",
            total_items=len(req.resources),
            started_at=datetime.utcnow()
        )
        session.add(job)
        await session.commit()
    
    metrics_logger.log_ingest_start(job_id, len(req.resources))
    logger.info("Ingestion job started",
               job_id=job_id,
               course_id=req.course_id,
               resources_count=len(req.resources))
    
    try:
        # Initialize services
        ingest_service = get_ingest_service()

        processed_count = 0
        failed_count = 0
        total_chunks = 0
        errors = []
        route = None

        # Fetch resources if provided
        documents = []
        if req.resources and len(req.resources) > 0:
            logger.info("Fetching resources", count=len(req.resources))
            documents = await fetch_resources(req.resources)

        if not documents:
            # Fallback: no resources available -> generate course route only (titles + descriptions), no content ingest
            logger.info("No resources available, generating course route via LLM (titles + descriptions only)")

            # Generate a course route (with local fallback)
            try:
                route = await generate_course_route(
                    req.course_id,
                    req.title,
                    req.description,
                    req.resources,
                    req.lang
                )
            except Exception:
                # Minimal fallback route if JSON generation/parse fails
                module_id = str(uuid.uuid4())
                lessons = []
                base_titles = [
                    "Introduction", "Core Concepts", "Hands-on Practice", "Advanced Topics", "Summary & Next Steps"
                ]
                for idx, t in enumerate(base_titles, start=1):
                    lessons.append({
                        "lesson_id": str(uuid.uuid4()),
                        "title": f"{t}",
                        "description": "",
                        "order": idx,
                        "min_mastery": 0.65
                    })
                route = {
                    "modules": [{
                        "module_id": module_id,
                        "title": req.title or "Module 1",
                        "description": req.description or "",
                        "order": 1,
                        "lessons": lessons
                    }]
                }
            # No ingestion of generated content; we only return the route in callback

        else:
            # Process fetched documents
            for i, doc in enumerate(documents):
                try:
                    # Update progress
                    async with async_session() as session:
                        job = await session.get(IngestJob, job_id)
                        if job:
                            job.processed_items = i + 1
                            await session.commit()

                    # Ingest document
                    chunks_count = await ingest_service.ingest_text(
                        course_id=req.course_id,
                        raw_text=doc.get("content", ""),
                        source_ref=doc.get("source"),
                        language=req.lang,
                        metadata={
                            "document_type": doc.get("document_type"),
                            "title": doc.get("title"),
                            **doc.get("metadata", {})
                        }
                    )

                    processed_count += 1
                    total_chunks += chunks_count

                    logger.info(
                        "Document processed",
                        document=doc.get("source"),
                        chunks_created=chunks_count,
                        progress=f"{i+1}/{len(documents)}"
                    )

                except Exception as e:
                    failed_count += 1
                    errors.append({
                        "document": doc.get("source", f"document_{i}"),
                        "error": str(e)
                    })
                    logger.error("Document processing failed", document=doc.get("source"), error=str(e))

            # Generate course route (normal path when resources exist)
            logger.info("Generating course route")
            try:
                route = await generate_course_route(
                    req.course_id,
                    req.title,
                    req.description,
                    req.resources,
                    req.lang
                )
            except Exception:
                # Minimal fallback route
                module_id = str(uuid.uuid4())
                lessons = []
                base_titles = [
                    "Introduction", "Core Concepts", "Hands-on Practice", "Advanced Topics", "Summary & Next Steps"
                ]
                for idx, t in enumerate(base_titles, start=1):
                    lessons.append({
                        "lesson_id": str(uuid.uuid4()),
                        "title": f"{t}",
                        "description": "",
                        "order": idx,
                        "min_mastery": 0.65
                    })
                route = {
                    "modules": [{
                        "module_id": module_id,
                        "title": req.title or "Module 1",
                        "description": req.description or "",
                        "order": 1,
                        "lessons": lessons
                    }]
                }

        # Pretty log the generated course before callback
        def _format_course_route(r: dict) -> str:
            modules = r.get("modules", []) or []
            lines: list[str] = []
            lines.append(f"Course: {req.title or ''} [{req.course_id}] lang={req.lang}")
            lines.append(f"Modules: {len(modules)}")
            for m in modules:
                m_pos = m.get("position", 1)
                m_title = m.get("title", "")
                m_id = m.get("module_id", "")
                lines.append(f"  {m_pos}. {m_title}  (module_id={m_id})")
                lessons = m.get("lessons", []) or []
                for l in lessons:
                    l_pos = l.get("position", 1)
                    l_title = l.get("title", "")
                    l_mm = l.get("min_mastery", 0.65)
                    l_id = l.get("lesson_id", "")
                    lines.append(f"    {l_pos}) {l_title}  mm={l_mm}  (lesson_id={l_id})")
            return "\n".join(lines)

        try:
            preview = _format_course_route(route)
            logger.info(
                "generated_course_route",
                course_id=req.course_id,
                lang=req.lang,
                modules=len(route.get("modules", []) or []),
                preview=preview
            )
        except Exception as _:
            # Fallback structured summary if formatting ever fails
            logger.info(
                "generated_course_route",
                course_id=req.course_id,
                lang=req.lang,
                modules=len(route.get("modules", []) or [])
            )
        
        # Update job record
        async with async_session() as session:
            job = await session.get(IngestJob, job_id)
            if job:
                job.status = "completed"
                job.processed_items = processed_count
                job.failed_items = failed_count
                job.chunks_created = total_chunks
                job.completed_at = datetime.utcnow()
                job.result_data = {
                    "route": route,
                    "errors": errors
                }
                await session.commit()
        
        # Prepare success result
        duration = time.time() - start_time
        metrics_logger.log_ingest_complete(job_id, duration, total_chunks)
        
        # Build callback strictly following provided schema
        result = {
            "job_id": job_id,
            "course_id": req.course_id,
            "status": "success",
            "description": f"Ingest completed successfully ({len(req.resources)} resources).",
            "lang": req.lang,
            "route": route
        }
        
    except Exception as e:
        # Update job record
        async with async_session() as session:
            job = await session.get(IngestJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                await session.commit()
        
        metrics_logger.log_ingest_error(job_id, str(e))
        logger.error("Ingestion job failed",
                    job_id=job_id,
                    error=str(e),
                    error_type=type(e).__name__)
        
        result = {
            "job_id": job_id,
            "course_id": req.course_id,
            "status": "failed",
            "description": str(e),
            "lang": req.lang,
            "route": None
        }
    
    # Send callback
    try:
        await send_callback(result)
    except Exception as e:
        logger.error("Callback failed",
                    job_id=job_id,
                    error=str(e))
        # Don't fail the job if callback fails
    
    # Clear job context
    job_id_var.set(None)
