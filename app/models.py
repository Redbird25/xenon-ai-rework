import uuid
from sqlalchemy import Column, Integer, Text, String, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class LessonChunk(Base):
    __tablename__ = "lesson_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(UUID(as_uuid=True), nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(768))
    source_ref = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
