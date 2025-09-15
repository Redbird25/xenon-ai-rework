from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field

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


# ======================= Quiz Schemas =======================

class UserPref(BaseModel):
    interests: List[str] = Field(default_factory=list)
    hobbies: List[str] = Field(default_factory=list)
    learning_style: str = Field(default="TEXT")


class QuizGenerateRequest(BaseModel):
    # Identification
    lesson_material_id: str
    # Optional external quiz identifier from Core (echoed back in callback)
    quiz_id: Optional[str] = Field(default=None, validation_alias="quizId")
    # New topic-based inputs from Core
    title: str
    description: Optional[str] = None
    # Preferences and knobs
    user_pref: UserPref
    course_id: Optional[str] = None  # ignored for topic-based search
    question_count: int = Field(default=10, ge=1, le=50)
    open_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    mcq_multi_allowed: bool = Field(default=True)


class QuizGenerateResponse(BaseModel):
    status: str
    job_id: str


class QuizOption(BaseModel):
    id: str
    text: str


class QuizQuestion(BaseModel):
    id: str
    type: Literal['open', 'short_answer', 'mcq_single', 'mcq_multi']
    prompt: str
    # Closed-form fields
    options: Optional[List[QuizOption]] = None
    correct_option_ids: Optional[List[str]] = None
    # Open-form fields
    acceptable_answers: Optional[List[str]] = None
    acceptable_keywords: Optional[List[List[str]]] = None
    # Common fields
    source_chunk_ids: List[int] = Field(default_factory=list)
    difficulty: Optional[Literal['easy','medium','hard']] = None
    explanation: Optional[str] = None


class QuizContent(BaseModel):
    quiz_id: str
    language: str = Field(default="en")
    questions: List[QuizQuestion]
    meta: Dict[str, Any] = Field(default_factory=dict)


class QuizCallbackPayload(BaseModel):
    job_id: str
    lesson_material_id: str
    quiz_id: Optional[str] = Field(default=None, serialization_alias="quizId")
    status: Literal['success','failed']
    description: str
    content: Optional[QuizContent] = None


# Evaluation
class UserAnswer(BaseModel):
    question_id: str
    # For mcq: list of option ids; for open/short_answer: a single string
    answer: Union[str, List[str]]


class QuestionVerdict(BaseModel):
    question_id: str
    verdict: Literal['correct','partial','incorrect']
    score: float = Field(ge=0.0, le=1.0)
    explanation: Optional[str] = None


class QuizEvaluateResponse(BaseModel):
    quiz_id: str
    score_percent: float = Field(ge=0.0, le=100.0)
    details: List[QuestionVerdict]


# Minimal evaluate payload items: only question text and user's answer
class QAItem(BaseModel):
    question: str
    # Always send answers as an array, even for open/short
    answer: List[str]




class QuizEvaluateByLessonRequest(BaseModel):
    lesson_material_id: str
    items: List[QAItem]


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
