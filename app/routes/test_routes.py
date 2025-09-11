"""
Test routes for lesson materialization demo
"""
from fastapi import APIRouter
from app.schemas import MaterializeLessonRequest, MaterializedLesson, LessonSection
from app.core.logging import get_logger
import uuid

logger = get_logger(__name__)

router = APIRouter(prefix="/ai/test", tags=["test"])


@router.post("/lesson-demo")
async def demo_lesson_materialization(req: MaterializeLessonRequest):
    """
    Demo endpoint to show lesson materialization results without actual processing
    """
    job_id = str(uuid.uuid4())
    
    # Simulate different strategies based on lesson name
    if "variables" in req.lesson_name.lower():
        strategy = "resource_none"
        chunks_used = 0
        chunks_created = 5
        should_ingest = True
    elif "python" in req.lesson_name.lower():
        strategy = "resource_mixed"
        chunks_used = 2
        chunks_created = 2
        should_ingest = True
    else:
        strategy = "resource_rich"
        chunks_used = 8
        chunks_created = 0
        should_ingest = False
    
    # Create mock lesson
    sections = [
        LessonSection(
            title="Introduction",
            content="This is the introduction section",
            examples=["Example 1", "Example 2"]
        ),
        LessonSection(
            title="Core Concepts",
            content="Main concepts explained here",
            examples=["Practical example", "Real-world scenario"]
        ),
        LessonSection(
            title="Advanced Topics",
            content="Advanced material covered",
            examples=["Complex example"]
        )
    ]
    
    mock_lesson = MaterializedLesson(
        lesson_name=req.lesson_name,
        description=req.description,
        sections=sections,
        generated_from_chunks=list(range(chunks_used))
    )
    mock_lesson._content_strategy = strategy
    
    # 🎨 Beautiful callback result logging (same as production)
    processing_time = 12.5
    strategy_emoji = {
        "resource_rich": "📚",
        "resource_mixed": "🔄", 
        "resource_none": "🆕"
    }.get(strategy, "❓")
    
    # Add note about embedding quota issue
    quota_note = ""
    if should_ingest:
        quota_note = " (Note: Demo mode - no actual embeddings created due to API quotas)"
    
    logger.info(
        "🎓 LESSON MATERIALIZATION COMPLETED" + quota_note,
        job_id=job_id,
        lesson_name=mock_lesson.lesson_name,
        extra={
            "STRATEGY": f"{strategy_emoji} {strategy.upper()}",
            "SECTIONS": f"📝 {len(mock_lesson.sections)} sections generated",
            "CHUNKS_USED": f"🔍 {chunks_used} source chunks",
            "CONTENT_INGESTED": f"{'✅' if should_ingest else '❌'} {'YES (DEMO)' if should_ingest else 'NO'}",
            "CHUNKS_CREATED": f"➕ {chunks_created} new chunks (simulated)",
            "PROCESSING_TIME": f"⏱️ {processing_time:.1f}s",
            "USER_STYLE": f"🎯 {req.user_pref.learning_style}",
            "INTERESTS": f"❤️ {', '.join(req.user_pref.interests[:3])}" + ("..." if len(req.user_pref.interests) > 3 else ""),
            "API_STATUS": "🚫 Embedding API quota exceeded - using demo mode"
        }
    )
    
    # Return callback-style result
    return {
        "job_id": job_id,
        "course_id": req.course_id,
        "status": "completed",
        "lesson_data": {
            "lesson_name": mock_lesson.lesson_name,
            "description": mock_lesson.description,
            "sections": [section.model_dump() for section in mock_lesson.sections],
            "generated_from_chunks": mock_lesson.generated_from_chunks,
            "content_strategy": strategy
        },
        "processing_time_seconds": processing_time,
        "chunks_created": chunks_created,
        "content_ingested": should_ingest,
        "strategy_analysis": {
            "strategy": strategy,
            "reasoning": {
                "resource_rich": "Found sufficient resources (8+ chunks), using existing materials only",
                "resource_mixed": "Limited resources found (2 chunks), mixing with LLM generation", 
                "resource_none": "No relevant resources found, full LLM generation + ingestion"
            }.get(strategy),
            "chunks_analysis": {
                "found": chunks_used,
                "high_relevance": min(chunks_used, 2),
                "created": chunks_created
            }
        }
    }