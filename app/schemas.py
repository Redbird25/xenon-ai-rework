from pydantic import BaseModel

class IngestRequest(BaseModel):
    course_id: str
    title: str
    description: str | None = None
    resources: list[str]
    lang: str


class IngestResponse(BaseModel):
    status: str
    job_id: str
