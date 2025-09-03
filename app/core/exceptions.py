"""
Custom exceptions and error handling for AI Ingest Service
"""
from typing import Optional, Dict, Any


class AIIngestException(Exception):
    """Base exception for AI Ingest Service"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentProcessingError(AIIngestException):
    """Error during document processing"""
    pass


class EmbeddingGenerationError(AIIngestException):
    """Error during embedding generation"""
    pass


class LLMError(AIIngestException):
    """Error during LLM operations"""
    pass


class ChunkingError(AIIngestException):
    """Error during document chunking"""
    pass


class VectorSearchError(AIIngestException):
    """Error during vector search operations"""
    pass


class ResourceFetchError(AIIngestException):
    """Error fetching external resources"""
    pass


class RateLimitError(AIIngestException):
    """Rate limit exceeded"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message, {"retry_after": retry_after})
        self.retry_after = retry_after


class ValidationError(AIIngestException):
    """Input validation error"""
    pass


class ConfigurationError(AIIngestException):
    """Configuration error"""
    pass


class DatabaseError(AIIngestException):
    """Database operation error"""
    pass
