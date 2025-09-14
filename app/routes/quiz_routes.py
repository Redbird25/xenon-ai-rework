from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.schemas import (
    QuizGenerateRequest,
    QuizEvaluateResponse, QuizEvaluateByLessonRequest,
    QuizCallbackPayload
)
from app.quiz_job import run_quiz_job
from app.core.quiz import AnswerEvaluator, QuizGenerator, select_chunk_ids_for_topic, sanitize_quiz_for_delivery
from app.core.logging import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/ai/quiz", tags=["quiz"])


@router.post("/generate", response_model=QuizCallbackPayload)
async def generate_quiz(req: QuizGenerateRequest, background: BackgroundTasks):
    try:
        import uuid as _uuid
        job_id = str(_uuid.uuid4())

        # Generate synchronously and return the exact JSON payload
        generator = QuizGenerator()
        # Determine chunk ids by title/description/language
        chunk_ids = await select_chunk_ids_for_topic(
            title=req.title,
            description=req.description,
            language=None,
            top_k=max(20, int(getattr(req, 'question_count', 10)) * 4)
        )
        topic_context = None
        if not chunk_ids:
            topic_context = (f"Topic: {req.title}\n\nDescription: {req.description or ''}").strip()
        quiz = await generator.generate_quiz(
            chunk_ids=chunk_ids,
            user_pref=req.user_pref.model_dump() if hasattr(req.user_pref, 'model_dump') else dict(req.user_pref),
            question_count=getattr(req, 'question_count', 10),
            open_ratio=getattr(req, 'open_ratio', 0.4),
            mcq_multi_allowed=getattr(req, 'mcq_multi_allowed', True),
            language_override=None,
            topic_context=topic_context,
            topic_title=req.title,
            topic_description=req.description
        )
        payload = QuizCallbackPayload(
            job_id=job_id,
            lesson_material_id=req.lesson_material_id,
            status='success',
            description='Quiz generated successfully.',
            # Only questions and options for Core/front
            content=sanitize_quiz_for_delivery(quiz)  # Pydantic will coerce from dict via model type
        )
        return payload
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
