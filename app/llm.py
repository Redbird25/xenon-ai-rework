from google import genai
from google.genai import types
import os, uuid, json

# Client initialization with stable API v1
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version='v1')
)

async def generate_course_route(course_id: str, title: str, description: str, resources: list[str]):
    # Structured prompt using types.Content
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(
            f"Course ID: {course_id}\n"
            f"Title: {title}\n"
            f"Description: {description}\n"
            f"Resources: {resources}\n"
            f"Generate JSON route: modules with lessons (include lesson_id UUID, title, description, order, min_mastery=0.65)"
        )]
    )
    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=1000,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[user_content],
        config=config
    )

    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        return {
            "modules": [
                {
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
