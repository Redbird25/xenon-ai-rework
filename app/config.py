"""
Modern configuration using Pydantic Settings for AI Ingest Service
"""
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"


class Settings(BaseSettings):
    """Application settings with validation and type safety"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@75.119.145.146:5433/xenon_ai_db",
        description="PostgreSQL connection URL with pgvector support"
    )
    database_pool_size: int = Field(default=10)
    database_echo: bool = Field(default=False)
    
    # API Keys
    gemini_api_key: str = Field(
        default="AIzaSyBAkY3izlayY0oQtbEk46_tz7Bss0fQLd8",
        description="Google Gemini API key"
    )
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    
    # LLM Configuration
    llm_model: str = Field(default="gemini-2.5-flash")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=4096)
    llm_timeout: int = Field(default=60)
    llm_max_retries: int = Field(default=3)
    
    # Embedding Configuration
    embedding_model: str = Field(default="models/embedding-001")
    embedding_dim: int = Field(default=768)
    embedding_batch_size: int = Field(default=32)
    max_embedding_length: int = Field(default=8192)
    
    # Chunking Configuration
    chunking_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.RECURSIVE)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    min_chunk_size: int = Field(default=100)
    
    # Vector Search
    vector_search_k: int = Field(default=10)
    similarity_threshold: float = Field(default=0.7)
    
    # External Services
    core_callback_url: str = Field(
        default="http://75.119.145.146:8082/api/course/ingest/callback",
        description="URL for job completion callbacks"
    )
    
    # Rate Limiting
    rate_limit_rpm: int = Field(default=60)
    rate_limit_tpm: int = Field(default=90000)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    
    @field_validator("chunk_overlap")
    def validate_chunk_overlap(cls, v, info):
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    @field_validator("llm_temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return v
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT


# Create global settings instance
settings = Settings()

# For backward compatibility
DATABASE_URL = settings.database_url
GEMINI_API_KEY = settings.gemini_api_key
EMBEDDING_DIM = settings.embedding_dim
CORE_CALLBACK_URL = settings.core_callback_url
