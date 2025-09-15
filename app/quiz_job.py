"""
Asynchronous quiz generation job with callback
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Any
import httpx
import json

from app.config import settings
from app.core.quiz import QuizGenerator, select_chunk_ids_for_topic, sanitize_quiz_for_delivery
from app.core.cache import get_cache
from app.core.logging import get_logger, metrics_logger, job_id_var
from app.core.retry import RetryWithBackoff

logger = get_logger(__name__)


callback_retry = RetryWithBackoff(max_attempts=3, initial_delay=1.0)


@callback_retry
async def send_quiz_callback(callback_url: str, result: Dict[str, Any]):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(callback_url, json=result)
        resp.raise_for_status()
        logger.info("Quiz callback sent", job_id=result.get("job_id"), status=result.get("status"))


async def run_quiz_job(req, job_id: str):
    start = time.time()
    job_id_var.set(job_id)

    logger.info(
        "Quiz job started",
        job_id=job_id,
        lesson_material_id=getattr(req, 'lesson_material_id', None),
        chunk_count=len(getattr(req, 'generated_chunks', []) or [])
    )

    try:
        generator = QuizGenerator()
        # Determine chunk_ids: prefer explicit; else select by title/description/language
        chunk_ids = await select_chunk_ids_for_topic(
            title=getattr(req, 'title', ''),
            description=getattr(req, 'description', None),
            language=None,
            top_k=max(20, int(getattr(req, 'question_count', 10)) * 4)
        )
        # Build topic context if still nothing was found
        topic_context = None
        if not chunk_ids:
            title = getattr(req, 'title', '') or ''
            desc = getattr(req, 'description', '') or ''
            topic_context = (f"Topic: {title}\n\nDescription: {desc}").strip()
        quiz = await generator.generate_quiz(
            chunk_ids=chunk_ids,
            user_pref=req.user_pref.model_dump() if hasattr(req.user_pref, 'model_dump') else dict(req.user_pref),
            question_count=getattr(req, 'question_count', 10),
            open_ratio=getattr(req, 'open_ratio', 0.4),
            mcq_multi_allowed=getattr(req, 'mcq_multi_allowed', True),
            language_override=None,
            topic_context=topic_context,
            topic_title=getattr(req, 'title', None),
            topic_description=getattr(req, 'description', None)
        )

        # Ensure internal quiz_id matches external quizId from Core if provided
        try:
            external_quiz_id = getattr(req, 'quiz_id', None)
            if external_quiz_id:
                quiz["quiz_id"] = external_quiz_id
        except Exception:
            pass

        # Store full quiz spec for evaluation by lesson_material_id (Redis TTL)
        try:
            cache = get_cache()
            await cache.set_json(
                key=f"quiz:lesson:{req.lesson_material_id}",
                value={
                    "lesson_material_id": req.lesson_material_id,
                    "quiz_id": quiz.get("quiz_id"),
                    "language": quiz.get("language"),
                    "questions": quiz.get("questions", []),
                    "meta": quiz.get("meta", {})
                }
            )
        except Exception as _:
            # Cache is optional; continue even if it fails
            pass

        result = {
            "job_id": job_id,
            "lesson_material_id": req.lesson_material_id,
            "quizId": getattr(req, 'quiz_id', None),
            "status": "success",
            "description": "Quiz generated successfully.",
            # Only questions and options for Core
            "content": sanitize_quiz_for_delivery(quiz)
        }

    except Exception as e:
        logger.error("Quiz generation failed", job_id=job_id, error=str(e))
        result = {
            "job_id": job_id,
            "lesson_material_id": getattr(req, 'lesson_material_id', None),
            "quizId": getattr(req, 'quiz_id', None),
            "status": "failed",
            "description": str(e),
            "content": None
        }

    # Send callback or print JSON if no valid callback_url
    # Prefer request callback if valid else fallback to env
    callback_url = getattr(req, 'callback_url', None) or settings.materialization_quiz_callback_url
    if callback_url and isinstance(callback_url, str) and callback_url.lower().startswith(("http://","https://")):
        try:
            await send_quiz_callback(callback_url, result)
        except Exception as e:
            logger.error("Quiz callback failed", job_id=job_id, error=str(e))
    else:
        # No valid callback_url provided — print exact JSON payload to logs
        try:
            logger.info("quiz_callback_payload", payload=result)
        except Exception:
            # Fallback plain printing
            print(json.dumps(result, ensure_ascii=False))

    job_id_var.set(None)
