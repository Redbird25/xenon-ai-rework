"""
Modern database models with support for vector search and metadata
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, Integer, Text, String, DateTime, Float, JSON, 
    Index, func, text, and_, or_
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.hybrid import hybrid_property
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class LessonChunk(Base):
    """Document chunk with vector embeddings and metadata"""
    __tablename__ = "lesson_chunks"
    
    # Primary fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(), nullable=True)  # Dynamic dimension
    
    # Reference fields
    source_ref = Column(String(500), nullable=True)
    document_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Metadata as JSONB for flexibility (attribute named 'meta' to avoid SQLAlchemy reserved name)
    meta = Column('metadata', JSONB, nullable=True, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Search optimization fields
    embedding_model = Column(String(100), nullable=True)
    chunk_index = Column(Integer, nullable=True)
    chunk_size = Column(Integer, nullable=True)
    
    # Full-text search
    search_vector = Column(Text, nullable=True)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_course_chunks', 'course_id', 'chunk_index'),
        Index('idx_chunk_metadata', 'metadata', postgresql_using='gin'),
        Index('idx_chunk_created', 'created_at'),
        Index('idx_chunk_search_vector', 'search_vector', postgresql_using='gin'),
    )
    
    @hybrid_property
    def language(self) -> Optional[str]:
        """Get language from metadata"""
        return self.meta.get('language') if self.meta else None
    
    @hybrid_property
    def headers(self) -> Optional[Dict[str, str]]:
        """Get headers from metadata"""
        return self.meta.get('headers') if self.meta else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'course_id': str(self.course_id) if self.course_id else None,
            'chunk_text': self.chunk_text,
            'source_ref': self.source_ref,
            'document_id': str(self.document_id) if self.document_id else None,
            'metadata': self.meta or {},
            'embedding_model': self.embedding_model,
            'chunk_index': self.chunk_index,
            'chunk_size': self.chunk_size,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Document(Base):
    """Source document metadata"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Document info
    title = Column(String(500), nullable=True)
    source_url = Column(Text, nullable=True)
    document_type = Column(String(50), nullable=True)  # pdf, html, markdown, etc.
    language = Column(String(10), nullable=True)
    
    # Processing metadata
    processing_status = Column(String(50), default='pending')  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    chunks_count = Column(Integer, default=0)
    
    # Content hash for deduplication
    content_hash = Column(String(64), nullable=True, unique=True)
    
    # Additional metadata (attribute named 'meta' to avoid reserved name)
    meta = Column('metadata', JSONB, nullable=True, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index('idx_document_course', 'course_id'),
        Index('idx_document_status', 'processing_status'),
        Index('idx_document_hash', 'content_hash'),
    )


class SearchQuery(Base):
    """Track search queries for analytics and improvement"""
    __tablename__ = "search_queries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Query details
    query_text = Column(Text, nullable=False)
    enhanced_query = Column(Text, nullable=True)
    query_embedding = Column(Vector(), nullable=True)
    
    # Search parameters
    search_params = Column(JSONB, nullable=True, default={})
    filters = Column(JSONB, nullable=True, default={})
    
    # Results metadata
    results_count = Column(Integer, nullable=True)
    top_score = Column(Float, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    
    # User feedback
    user_rating = Column(Integer, nullable=True)  # 1-5 rating
    user_feedback = Column(Text, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_query_course', 'course_id'),
        Index('idx_query_created', 'created_at'),
    )


class IngestJob(Base):
    """Track ingestion jobs"""
    __tablename__ = "ingest_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Job details
    status = Column(String(50), default='pending')  # pending, processing, completed, failed
    job_type = Column(String(50), nullable=True)  # document, batch, api
    
    # Progress tracking
    total_items = Column(Integer, nullable=True)
    processed_items = Column(Integer, default=0)
    failed_items = Column(Integer, default=0)
    
    # Results
    chunks_created = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    result_data = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index('idx_job_course', 'course_id'),
        Index('idx_job_status', 'status'),
        Index('idx_job_created', 'created_at'),
    )
