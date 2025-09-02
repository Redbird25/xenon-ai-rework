import httpx
import uuid
from app.config import CORE_CALLBACK_URL
from app.ingest import ingest_text
from app.utils.resources import fetch_resource
from app.llm import generate_course_route

async def run_ingest_job(req, job_id: str):
    try:
        # Инжест ресурсов
        for resource in req.resources:
            text = await fetch_resource(resource)
            await ingest_text(req.course_id, text, resource)

        # Генерация маршрута
        route = await generate_course_route(
            req.course_id,
            req.title,
            req.description,
            req.resources,
            req.lang
        )

        # 🔒 Safety net: гарантируем наличие module_id и lesson_id
        for module in route.get("modules", []):
            if "module_id" not in module:
                module["module_id"] = str(uuid.uuid4())
            for lesson in module.get("lessons", []):
                if "lesson_id" not in lesson:
                    lesson["lesson_id"] = str(uuid.uuid4())
                if "min_mastery" not in lesson:
                    lesson["min_mastery"] = 0.65

        result = {
            "job_id": job_id,
            "course_id": req.course_id,
            "status": "success",
            "description": f"Ingest completed successfully ({len(req.resources)} resources).",
            "route": route,
            "lang": req.lang
        }

    except Exception as e:
        result = {
            "job_id": job_id,
            "course_id": req.course_id,
            "status": "failed",
            "description": str(e),
            "route": None,
            "lang": req.lang
        }

    # ✅ Отправляем результат в Core по фиксированному callback
    async with httpx.AsyncClient() as client:
        await client.post(CORE_CALLBACK_URL, json=result)
