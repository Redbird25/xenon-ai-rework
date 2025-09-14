from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.schemas import (
    QuizGenerateRequest, QuizGenerateResponse,
    QuizEvaluateResponse, QuizEvaluateByLessonRequest,
)
from app.quiz_job import run_quiz_job
from app.core.quiz import AnswerEvaluator
from app.core.cache import get_cache
from app.config import settings
from app.core.logging import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/ai/quiz", tags=["quiz"])


@router.post("/generate", response_model=QuizGenerateResponse)
async def generate_quiz(req: QuizGenerateRequest, background: BackgroundTasks):
    try:
        import uuid as _uuid
        job_id = str(_uuid.uuid4())
        # Orchestrate with Redis pending/ready flags
        cache = get_cache()
        ready_key = f"lesson:ready:{req.lesson_material_id}"
        pending_key = f"quiz:pending:{req.lesson_material_id}"

        if await cache.exists(ready_key):
            background.add_task(run_quiz_job, req, job_id)
        else:
            await cache.set_json(
                pending_key,
                {"job_id": job_id, "request": req.model_dump(by_alias=True)},
                ttl_seconds=settings.quiz_pending_ttl_seconds,
            )
        return QuizGenerateResponse(status="accepted", job_id=job_id)
    except Exception as e:
        logger.error("Failed to start quiz job", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start quiz job: {e}")


@router.post("/evaluate", response_model=QuizEvaluateResponse)
async def evaluate_quiz(req: QuizEvaluateByLessonRequest):
    try:
        evaluator = AnswerEvaluator()
        result = await evaluator.evaluate_by_lesson(
            lesson_material_id=req.lesson_material_id,
            items=[i.model_dump() for i in req.items]
        )
        return QuizEvaluateResponse(**result)
    except Exception as e:
        logger.error("Quiz evaluation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
