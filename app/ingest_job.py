import httpx
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
        route = await generate_course_route(req.course_id, req.title, req.description, req.resources, req.lang)

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

    # ❗ Отправляем результат прямо в Core (жёсткий callback)
    async with httpx.AsyncClient() as client:
        await client.post(CORE_CALLBACK_URL, json=result)
