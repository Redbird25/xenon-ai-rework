from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal


class IngestRequest(BaseModel):
    course_id: str
    title: str
    description: str | None = None
    resources: list[str]
    lang: str


class IngestResponse(BaseModel):
    status: str
    job_id: str


class UserPreferences(BaseModel):
    interests: List[str]
    hobbies: List[str]
    learning_style: Literal["TEXT", "VIDEO", "MIXED"]


class CodeExample(BaseModel):
    language: str
    code: str
    explanation: str
    context: str


class PracticalExercise(BaseModel):
    task: str
    solution_hint: str
    difficulty: str


class LessonSection(BaseModel):
    title: str
    content: str
    examples: List[str]
    code_examples: Optional[List[CodeExample]] = []
    practical_exercises: Optional[List[PracticalExercise]] = []


class MaterializedLesson(BaseModel):
    lesson_name: str
    description: str
    sections: List[LessonSection]
    generated_from_chunks: List[int]  # IDs of source chunks


class MaterializeLessonRequest(BaseModel):
    course_id: str
    lesson_name: str
    lesson_description: str
    user_pref: UserPreferences
    lesson_material_id: Optional[str] = None


class MaterializeLessonResponse(BaseModel):
    status: str
    job_id: str
