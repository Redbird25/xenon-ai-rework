"""
Lesson materialization routes
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.schemas import MaterializeLessonRequest, MaterializeLessonResponse
from app.lesson_job import run_lesson_materialization_job
from app.core.logging import get_logger
import uuid

logger = get_logger(__name__)

router = APIRouter(prefix="/ai/lessons", tags=["lessons"])


@router.post("/materialize", response_model=MaterializeLessonResponse)
async def materialize_lesson(
    req: MaterializeLessonRequest, 
    background: BackgroundTasks
):
    """
    Materialize a lesson from course content based on lesson name, description and user preferences.
    
    This endpoint:
    1. Searches for relevant chunks in the course content
    2. Generates a structured lesson using LLM with personalized examples
    3. Ingests the academic content back into the system (excluding personalized examples)
    4. Returns job_id for tracking progress
    """
    try:
        job_id = str(uuid.uuid4())
        
        logger.info(
            "Lesson materialization request received",
            job_id=job_id,
            course_id=req.course_id,
            lesson_name=req.lesson_name,
            user_learning_style=req.user_pref.learning_style
        )
        
        # Start background job
        background.add_task(
            run_lesson_materialization_job, 
            req, 
            job_id
        )
        
        return MaterializeLessonResponse(
            status="accepted", 
            job_id=job_id
        )
        
    except Exception as e:
        logger.error(
            "Failed to start lesson materialization",
            error=str(e),
            lesson_name=req.lesson_name
        )
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start lesson materialization: {str(e)}"
        )