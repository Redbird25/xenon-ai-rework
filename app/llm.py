from google import genai
from google.genai import types
import os, uuid, json

# Client initialization with stable API v1
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version='v1')
)

async def generate_course_route(course_id: str, title: str, description: str, resources: list[str], lang: str = "en"):
    # Structured prompt using types.Content
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(
            f"Course ID: {course_id}\n"
            f"Title: {title}\n"
            f"Description: {description}\n"
            f"Resources: {resources}\n"
            f"Language: {lang}\n"
            f"Generate JSON route: modules with lessons.\n"
            f"Each module must have: module_id (UUID), title, position.\n"
            f"Each lesson must have: lesson_id (UUID), title, description, position, min_mastery=0.65."
        )]
    )
    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=1200,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[user_content],
        config=config
    )

    try:
        route = json.loads(resp.text)

        # 🔒 Safety net: если в ответе нет module_id, подставляем UUID
        for module in route.get("modules", []):
            if "module_id" not in module:
                module["module_id"] = str(uuid.uuid4())
            for lesson in module.get("lessons", []):
                if "lesson_id" not in lesson:
                    lesson["lesson_id"] = str(uuid.uuid4())
                if "min_mastery" not in lesson:
                    lesson["min_mastery"] = 0.65

        return route

    except Exception:
        # fallback — минимальный маршрут
        return {
            "modules": [
                {
                    "module_id": str(uuid.uuid4()),
                    "title": "Module 1",
                    "position": 1,
                    "lessons": [
                        {
                            "lesson_id": str(uuid.uuid4()),
                            "title": "Introduction",
                            "description": "Auto-gen intro",
                            "position": 1,
                            "min_mastery": 0.65
                        }
                    ]
                }
            ]
        }
