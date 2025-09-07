"""
Legacy LLM module for backward compatibility
"""
import uuid
from app.core.llm import get_course_generator


async def generate_course_route(course_id: str, title: str, description: str, resources: list[str], lang: str = "en"):
    """
    Legacy function for generating course routes.
    Uses the new CourseRouteGenerator under the hood.
    """
    generator = get_course_generator()
    
    # Generate route using new system
    route = await generator.generate_route(
        course_id=course_id,
        title=title,
        description=description,
        resources=resources,
        language=lang
    )
    
    # Ensure compatibility with legacy format
    for module in route.get("modules", []):
        if "module_id" not in module:
            module["module_id"] = str(uuid.uuid4())
        for lesson in module.get("lessons", []):
            if "lesson_id" not in lesson:
                lesson["lesson_id"] = str(uuid.uuid4())
            if "min_mastery" not in lesson:
                lesson["min_mastery"] = 0.65
    
    return route


async def generate_lesson_content(
    course_title: str,
    course_description: str,
    lesson_title: str,
    lang: str = "en"
) -> str:
    """Legacy wrapper to generate standalone lesson content via core LLM provider."""
    generator = get_course_generator()
    return await generator.generate_lesson_content(
        course_title=course_title,
        course_description=course_description,
        lesson_title=lesson_title,
        language=lang
    )
