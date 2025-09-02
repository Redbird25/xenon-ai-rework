import google.generativeai as genai
import os, uuid, json

# Настраиваем API ключ
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Создаём модель один раз
model = genai.GenerativeModel("gemini-2.5-flash")

async def generate_course_route(course_id: str, title: str, description: str, resources: list[str], lang: str = "en"):
    prompt = f"""
    Course ID: {course_id}
    Title: {title}
    Description: {description}
    Resources: {resources}
    Language: {lang}
    Generate JSON route: modules with lessons.
    Each module must have: module_id (UUID), title, order.
    Each lesson must have: lesson_id (UUID), title, description, order, min_mastery=0.65.
    """

    resp = model.generate_content(prompt)

    try:
        route = json.loads(resp.text)
        # safety net
        for module in route.get("modules", []):
            module.setdefault("module_id", str(uuid.uuid4()))
            for lesson in module.get("lessons", []):
                lesson.setdefault("lesson_id", str(uuid.uuid4()))
                lesson.setdefault("min_mastery", 0.65)
        return route
    except Exception:
        return {
            "modules": [
                {
                    "module_id": str(uuid.uuid4()),
                    "title": "Module 1",
                    "order": 1,
                    "lessons": [
                        {
                            "lesson_id": str(uuid.uuid4()),
                            "title": "Introduction",
                            "description": "Auto-gen intro",
                            "order": 1,
                            "min_mastery": 0.65
                        }
                    ]
                }
            ]
        }
