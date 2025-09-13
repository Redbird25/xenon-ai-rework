"""
Lesson materialization job processing
"""
import httpx
import uuid
import time
import asyncio
from datetime import datetime
from typing import Dict, Any

from app.config import settings
from app.core.logging import get_logger, job_id_var
from app.core.retry import RetryWithBackoff
from app.core.lesson_generator import get_lesson_generator
from app.ingest import get_ingest_service
from app.db import async_session
from app.models import IngestJob, MaterializedLesson
from app.schemas import MaterializeLessonRequest
from sqlalchemy import select

logger = get_logger(__name__)

# Retry decorator for callback
callback_retry = RetryWithBackoff(max_attempts=3, initial_delay=1.0)


@callback_retry
async def send_callback(result: Dict[str, Any]):
    """Send callback to core service with retry and error handling"""
    job_id = result.get("job_id", "unknown")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(settings.materialization_callback_url, json=result)
            response.raise_for_status()
            logger.info("Callback sent successfully",
                       job_id=job_id,
                       status=result.get("status"))
    except httpx.TimeoutException:
        logger.error(f"Callback timeout for job {job_id}")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"Callback HTTP error for job {job_id}: {e.response.status_code}")
        raise
    except Exception as e:
        logger.error(f"Callback failed for job {job_id}: {str(e)}")
        raise


async def log_beautiful_lesson(materialized_lesson, req: MaterializeLessonRequest, processing_time: float, chunks_created: int):
    """Write the materialized lesson content beautifully to a text file"""
    
    # Extract lesson content safely
    content = getattr(materialized_lesson, 'content', None) or {}
    sections = content.get('sections', []) if isinstance(content, dict) else []
    
    # Generate filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lesson_name = getattr(materialized_lesson, 'lesson_name', 'Unknown_Lesson')
    # Clean filename
    clean_name = "".join(c for c in lesson_name if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
    filename = f"lesson_{timestamp}_{clean_name.replace(' ', '_')}.md"
    
    # Build content
    output_lines = []
    
    # Markdown Header
    output_lines.append(f"# 📚 {lesson_name}")
    output_lines.append("")
    
    # Metadata section
    strategy = getattr(materialized_lesson, '_content_strategy', 'unknown')
    strategy_emoji = {
        "resource_rich": "📚 Resource-rich",
        "resource_mixed": "🔄 Mixed content", 
        "resource_none": "🆕 Standalone"
    }.get(strategy, "❓ Unknown")
    
    output_lines.append("## 📋 Lesson Metadata")
    output_lines.append("")
    output_lines.append(f"- **Strategy**: {strategy_emoji}")
    output_lines.append(f"- **Learning Style**: {req.user_pref.learning_style}")
    output_lines.append(f"- **Interests**: {', '.join(req.user_pref.interests)}")
    hobbies = getattr(req.user_pref, 'hobbies', []) or []
    if hobbies:
        output_lines.append(f"- **Hobbies**: {', '.join(hobbies)}")
    output_lines.append(f"- **Generated in**: {processing_time:.1f}s")
    output_lines.append(f"- **Sections**: {len(sections)}")
    source_chunks = getattr(materialized_lesson, 'generated_from_chunks', []) or []
    output_lines.append(f"- **Source chunks**: {len(source_chunks)}")
    output_lines.append(f"- **Chunks created**: {chunks_created}")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    
    # Content sections
    for i, section in enumerate(sections, 1):
        section_title = section.get('title', f'Section {i}')
        section_content = section.get('content', '')
        section_examples = section.get('examples', [])
        code_examples = section.get('code_examples', [])
        practical_exercises = section.get('practical_exercises', [])

        output_lines.append(f"## 📖 {section_title}")
        output_lines.append("")
        output_lines.append(section_content.strip())
        output_lines.append("")

        # Add code examples for technical lessons (updated for both dict and Pydantic models)
        if code_examples and len(code_examples) > 0:
            output_lines.append("### 💻 Code Examples")
            output_lines.append("")
            for j, code_example in enumerate(code_examples, 1):
                # Handle both dictionary and Pydantic object formats
                if isinstance(code_example, dict):
                    context = code_example.get('context', 'General example')
                    language = code_example.get('language', 'text')
                    code = code_example.get('code', '')
                    explanation = code_example.get('explanation', '')
                else:
                    context = code_example.context if hasattr(code_example, 'context') else 'General example'
                    language = code_example.language if hasattr(code_example, 'language') else 'text'
                    code = code_example.code if hasattr(code_example, 'code') else ''
                    explanation = code_example.explanation if hasattr(code_example, 'explanation') else ''

                output_lines.append(f"#### Example {j}: {context}")
                output_lines.append("")
                output_lines.append(f"```{language}")
                output_lines.append(code.strip())
                output_lines.append("```")
                output_lines.append("")
                if explanation:
                    output_lines.append(f"**Explanation:** {explanation}")
                    output_lines.append("")

        # Add practical exercises for technical lessons (updated for both dict and Pydantic models)
        if practical_exercises and len(practical_exercises) > 0:
            output_lines.append("### 🏋️ Practical Exercises")
            output_lines.append("")
            for j, exercise in enumerate(practical_exercises, 1):
                # Handle both dictionary and Pydantic object formats
                if isinstance(exercise, dict):
                    task = exercise.get('task', '')
                    solution_hint = exercise.get('solution_hint', '')
                    difficulty = exercise.get('difficulty', 'intermediate')
                else:
                    task = exercise.task if hasattr(exercise, 'task') else ''
                    solution_hint = exercise.solution_hint if hasattr(exercise, 'solution_hint') else ''
                    difficulty = exercise.difficulty if hasattr(exercise, 'difficulty') else 'intermediate'

                difficulty_emoji = {
                    'beginner': '🟢',
                    'intermediate': '🟡',
                    'advanced': '🔴'
                }.get(difficulty, '🟡')

                output_lines.append(f"#### Exercise {j} {difficulty_emoji} ({difficulty.title()})")
                output_lines.append("")
                output_lines.append(f"**Task:** {task}")
                output_lines.append("")
                if solution_hint:
                    output_lines.append(f"💡 **Hint:** {solution_hint}")
                    output_lines.append("")

        # Add traditional examples if they exist
        if section_examples and len(section_examples) > 0:
            output_lines.append("### 💡 Additional Examples")
            output_lines.append("")
            for j, example in enumerate(section_examples, 1):
                output_lines.append(f"{j}. {example}")
            output_lines.append("")
    
    # If no sections, show full content
    if not sections and content:
        # Try to extract sections from various possible structures
        if 'sections' in content and content['sections']:
            sections = content['sections']
            output_lines.append("📖 SECTIONS FROM CONTENT:")
            output_lines.append("-" * 50)
            for i, section in enumerate(sections, 1):
                section_title = section.get('title', f'Section {i}')
                section_content = section.get('content', '')
                output_lines.append(f"📖 SECTION {i}: {section_title}")
                output_lines.append("-" * 30)
                output_lines.append(section_content.strip())
                output_lines.append("")
        else:
            full_content = str(content)
            if full_content and full_content != '{}':
                output_lines.append("📖 FULL CONTENT:")
                output_lines.append("-" * 50)
                output_lines.append(full_content)
                output_lines.append("")
    
    # Footer
    output_lines.append("---")
    output_lines.append("")
    output_lines.append("## ✅ Lesson Complete")
    output_lines.append("")
    output_lines.append("🎉 **Congratulations!** You've completed this personalized lesson.")
    output_lines.append("")
    output_lines.append("*Generated with AI Ingest Service - Personalized Learning System*")
    
    # Write to file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        logger.info(f"📄 Lesson saved to file: {filename}")
        logger.info(f"📝 Content lines: {len(output_lines)}")
        
    except Exception as e:
        logger.error(f"Failed to write lesson to file: {str(e)}")
        # Fallback to console output
        for line in output_lines:
            logger.info(line)


async def run_lesson_materialization_job(req: MaterializeLessonRequest, job_id: str):
    """
    Run lesson materialization job with progress tracking and robust error handling
    """
    start_time = time.time()

    # Global error handler to ensure callback is always sent
    try:
        job_id_var.set(job_id)

        # Initialize with safe logging
        try:
            logger.info(
                "Starting lesson materialization job",
                job_id=job_id,
                course_id=req.course_id,
                lesson_name=req.lesson_name,
                lesson_material_id=req.lesson_material_id
            )
        except Exception as log_error:
            # Fallback if logging fails
            print(f"Logger failed in job {job_id}: {str(log_error)}")
            print(f"Starting lesson materialization job for {req.lesson_name}")

        # Create job record
        async with async_session() as session:
            job = IngestJob(
                id=job_id,
                course_id=req.course_id,
                status="processing",
                job_type="lesson_materialization",
                total_items=1,
                started_at=datetime.utcnow()
            )
            session.add(job)
            await session.commit()

        # Main job processing
        # Generate lesson with additional error handling
        try:
            lesson_generator = get_lesson_generator()
            materialized_lesson = await lesson_generator.generate_lesson(
                course_id=req.course_id,
                lesson_name=req.lesson_name,
                description=req.lesson_description,
                user_preferences=req.user_pref,
                job_id=job_id
            )
        except Exception as gen_error:
            logger.error(f"Lesson generation failed: {str(gen_error)}")
            raise gen_error

        # Ingest academic content only if strategy requires it
        try:
            should_ingest, chunks_created = await ingest_lesson_content_conditionally(
                req.course_id, materialized_lesson, job_id
            )
        except Exception as ingest_error:
            logger.error(f"Content ingestion failed: {str(ingest_error)}")
            # Continue with lesson completion even if ingestion fails
            should_ingest, chunks_created = False, 0
        
        # Prepare callback data
        processing_time = time.time() - start_time
        result = {
            "job_id": job_id,
            "course_id": req.course_id,
            "lesson_material_id": req.lesson_material_id,
            "status": "completed",
            "processing_time_seconds": processing_time,
            "lessonData": {
                "lesson_name": materialized_lesson.lesson_name,
                "description": materialized_lesson.description,
                "sections": [
                    {
                        "title": section.get('title', '') if isinstance(section, dict) else getattr(section, 'title', ''),
                        "content": section.get('content', '') if isinstance(section, dict) else getattr(section, 'content', ''),
                        "examples": section.get('examples', []) if isinstance(section, dict) else getattr(section, 'examples', []),
                        "code_examples": [
                            {
                                "language": ce.get('language', '') if isinstance(ce, dict) else getattr(ce, 'language', ''),
                                "code": ce.get('code', '') if isinstance(ce, dict) else getattr(ce, 'code', ''),
                                "explanation": ce.get('explanation', '') if isinstance(ce, dict) else getattr(ce, 'explanation', ''),
                                "context": ce.get('context', '') if isinstance(ce, dict) else getattr(ce, 'context', '')
                            } for ce in (section.get('code_examples', []) if isinstance(section, dict) else getattr(section, 'code_examples', []))
                        ],
                        "practical_exercises": [
                            {
                                "task": ex.get('task', '') if isinstance(ex, dict) else getattr(ex, 'task', ''),
                                "solution_hint": ex.get('solution_hint', '') if isinstance(ex, dict) else getattr(ex, 'solution_hint', ''),
                                "difficulty": ex.get('difficulty', 'intermediate') if isinstance(ex, dict) else getattr(ex, 'difficulty', 'intermediate')
                            } for ex in (section.get('practical_exercises', []) if isinstance(section, dict) else getattr(section, 'practical_exercises', []))
                        ]
                    }
                    for section in (materialized_lesson.content.get('sections', []) if hasattr(materialized_lesson, 'content') and materialized_lesson.content else [])
                ],
                "generated_from_chunks": [str(chunk_id) for chunk_id in materialized_lesson.generated_from_chunks],
                "content_strategy": getattr(materialized_lesson, '_content_strategy', 'unknown')
            }
        }
        
        # 🎨 Beautiful callback result logging
        strategy = getattr(materialized_lesson, '_content_strategy', 'unknown')
        strategy_emoji = {
            "resource_rich": "📚",
            "resource_mixed": "🔄", 
            "resource_none": "🆕"
        }.get(strategy, "❓")
        
        logger.info(
            "🎓 LESSON MATERIALIZATION COMPLETED",
            job_id=job_id,
            lesson_name=materialized_lesson.lesson_name,
            extra={
                "STRATEGY": f"{strategy_emoji} {strategy.upper()}",
                "SECTIONS": f"📝 {len(materialized_lesson.content.get('sections', []) if hasattr(materialized_lesson, 'content') and materialized_lesson.content else [])} sections generated",
                "CHUNKS_USED": f"🔍 {len(materialized_lesson.generated_from_chunks)} source chunks",
                "CONTENT_INGESTED": f"{'✅' if should_ingest else '❌'} {'YES' if should_ingest else 'NO'}",
                "CHUNKS_CREATED": f"➕ {chunks_created} new chunks",
                "PROCESSING_TIME": f"⏱️ {processing_time:.1f}s",
                "USER_STYLE": f"🎯 {req.user_pref.learning_style}",
                "INTERESTS": f"❤️ {', '.join(req.user_pref.interests[:3])}" + ("..." if len(req.user_pref.interests) > 3 else "")
            }
        )
        
        # 📖 Beautiful lesson content display
        await log_beautiful_lesson(materialized_lesson, req, processing_time, chunks_created)
        
        # Update job status to completed first
        async with async_session() as session:
            await session.execute(
                IngestJob.__table__.update()
                .where(IngestJob.id == job_id)
                .values(
                    status="completed",
                    processed_items=1,
                    completed_at=datetime.utcnow()
                )
            )
            await session.commit()

        # Send success callback with multiple retry attempts
        callback_sent = False
        for attempt in range(3):
            try:
                await send_callback(result)
                logger.info("Callback sent successfully", job_id=job_id, status="completed", attempt=attempt+1)
                callback_sent = True
                break
            except Exception as e:
                logger.warning(f"Callback attempt {attempt+1} failed: {str(e)}")
                if attempt < 2:  # Don't wait after last attempt
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        if not callback_sent:
            logger.error("All callback attempts failed", job_id=job_id)
        
        logger.info(
            "Lesson materialization completed successfully",
            job_id=job_id,
            lesson_name=req.lesson_name,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(
            "Lesson materialization failed",
            job_id=job_id,
            error=error_msg,
            lesson_name=req.lesson_name
        )
        
        # Update job status to failed
        async with async_session() as session:
            await session.execute(
                IngestJob.__table__.update()
                .where(IngestJob.id == job_id)
                .values(
                    status="failed",
                    error_message=error_msg,
                    completed_at=datetime.utcnow()
                )
            )
            await session.commit()
        
        # Send failure callback
        result = {
            "job_id": job_id,
            "course_id": req.course_id,
            "lesson_material_id": req.lesson_material_id,
            "status": "failed",
            "processing_time_seconds": time.time() - start_time,
            "error": error_msg
        }
        
        # Send failure callback
        try:
            await send_callback(result)
            logger.info("Failure callback sent successfully", job_id=job_id, status="failed")
        except Exception as cb_e:
            logger.error("Failed to send failure callback", job_id=job_id, error=str(cb_e))
        
        # 💥 Log failure beautifully
        logger.error(
            "❌ LESSON MATERIALIZATION FAILED",
            job_id=job_id,
            lesson_name=req.lesson_name,
            error=error_msg,
            processing_time=f"{time.time() - start_time:.1f}s"
        )

    except Exception as critical_error:
        # Critical error handler - this catches everything including database errors
        error_msg = f"Critical job failure: {str(critical_error)}"
        print(f"CRITICAL ERROR in job {job_id}: {error_msg}")

        # Try to send failure callback even if database operations fail
        try:
            result = {
                "job_id": job_id,
                "course_id": req.course_id,
                "lesson_material_id": req.lesson_material_id,
                "status": "failed",
                "processing_time_seconds": time.time() - start_time,
                "error": error_msg
            }

            # Force callback send with basic HTTP call if everything else fails
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(settings.materialization_callback_url, json=result)
                print(f"Emergency callback sent for job {job_id}: {response.status_code}")
        except Exception as callback_error:
            print(f"Emergency callback also failed for job {job_id}: {str(callback_error)}")

        # Log to console as fallback
        print(f"Job {job_id} failed completely: {error_msg}")


async def ingest_lesson_content_conditionally(course_id: str, lesson: Any, job_id: str) -> tuple[bool, int]:
    """
    Conditionally ingest lesson content based on content strategy.
    
    Returns: (should_ingest, chunks_created)
    - resource_rich: не инжестим (0 chunks)
    - resource_mixed: инжестим только LLM-сгенерированные части
    - resource_none: полный инжест всего урока
    """
    
    # Get content strategy from lesson metadata
    content_strategy = 'resource_none'
    if hasattr(lesson, '_content_strategy'):
        content_strategy = lesson._content_strategy
    elif hasattr(lesson, 'content') and lesson.content:
        sections = lesson.content.get('sections', [])
        if sections and len(sections) > 0:
            content_strategy = sections[0].get('content_strategy', 'resource_none')
    
    logger.info(
        "Evaluating lesson content for ingestion",
        job_id=job_id,
        course_id=course_id,
        lesson_name=lesson.lesson_name,
        content_strategy=content_strategy
    )
    
    # Decision logic based on strategy
    if content_strategy == "resource_rich":
        # Много ресурсов - не инжестим
        logger.info(
            "Skipping ingestion - sufficient resources available",
            job_id=job_id,
            content_strategy=content_strategy
        )
        
        # Update record to show no ingestion needed
        async with async_session() as session:
            await session.execute(
                MaterializedLesson.__table__.update()
                .where(MaterializedLesson.job_id == job_id)
                .values(
                    is_ingested=0,  # 0 = no ingestion needed
                    ingested_chunks_count=0
                )
            )
            await session.commit()
        
        return False, 0
    
    elif content_strategy == "resource_mixed":
        # Мало ресурсов - инжестим только основной академический контент (без примеров)
        logger.info(
            "Starting mixed content ingestion - academic content only",
            job_id=job_id,
            content_strategy=content_strategy
        )
        
        try:
            # Extract only academic content (section titles and main content, no examples)
            academic_content = []
            
            sections = lesson.content.get('sections', []) if hasattr(lesson, 'content') and lesson.content else []
            for section in sections:
                # Add section title and main content, but exclude personalized examples
                section_title = section.get('title', 'Untitled Section')
                section_content = section.get('content', '')
                content_part = f"# {section_title}\n\n{section_content}"
                academic_content.append(content_part)
            
            # Combine all academic content
            full_academic_content = "\n\n".join(academic_content)
            
            chunks_created = await _ingest_content(
                course_id, lesson, job_id, full_academic_content, 
                content_strategy="mixed"
            )
            
            return True, chunks_created
            
        except Exception as e:
            logger.error(
                "Mixed content ingestion failed",
                job_id=job_id,
                error=str(e)
            )
            raise
    
    else:  # resource_none
        # Нет ресурсов - полный инжест всего урока
        logger.info(
            "Starting full content ingestion - no existing resources",
            job_id=job_id,
            content_strategy=content_strategy
        )
        
        try:
            # Extract full content including examples (будет использовано как база знаний)
            full_content = []
            
            sections = lesson.content.get('sections', []) if hasattr(lesson, 'content') and lesson.content else []
            for section in sections:
                section_title = section.get('title', 'Untitled Section')
                section_content = section.get('content', '')
                # Only include title and main content for search index, exclude examples
                content_part = f"# {section_title}\n\n{section_content}"
                full_content.append(content_part)
            
            # Combine all content
            full_lesson_content = "\n\n".join(full_content)
            
            chunks_created = await _ingest_content(
                course_id, lesson, job_id, full_lesson_content,
                content_strategy="full_generation"
            )
            
            return True, chunks_created
            
        except Exception as e:
            logger.error(
                "Full content ingestion failed",
                job_id=job_id,
                error=str(e)
            )
            raise


async def _ingest_content(course_id: str, lesson: Any, job_id: str, content: str, content_strategy: str) -> int:
    """Helper function to ingest content"""
    
    # Create a document-like structure for ingestion
    lesson_document = {
        "title": lesson.lesson_name,
        "content": content,
        "metadata": {
            "document_type": "generated_lesson",
            "lesson_name": lesson.lesson_name,
            "source": "lesson_materialization",
            "is_generated": True,
            "content_strategy": content_strategy,
            "generated_from_chunks": lesson.generated_from_chunks,
            "language": "en"  # Default, could be made configurable
        }
    }
    
    # Get ingest service and process the lesson content
    ingest_service = get_ingest_service()
    chunks_created = await ingest_service.ingest_text_content(
        course_id=course_id,
        content=lesson_document["content"],
        title=lesson_document["title"],
        metadata=lesson_document["metadata"]
    )
    
    # Update materialized lesson record
    async with async_session() as session:
        await session.execute(
            MaterializedLesson.__table__.update()
            .where(MaterializedLesson.job_id == job_id)
            .values(
                is_ingested=1,
                ingested_chunks_count=chunks_created
            )
        )
        await session.commit()
    
    logger.info(
        "Content ingestion completed",
        job_id=job_id,
        chunks_created=chunks_created,
        content_strategy=content_strategy
    )
    
    return chunks_created


async def get_lesson_chunks_count(job_id: str) -> int:
    """Get the number of chunks created for a lesson"""
    async with async_session() as session:
        result = await session.execute(
            select(MaterializedLesson.ingested_chunks_count)
            .where(MaterializedLesson.job_id == job_id)
        )
        chunks_count = result.scalar_one_or_none()
        return chunks_count or 0