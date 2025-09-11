"""
Modern document ingestion module with advanced RAG capabilities
"""
from typing import List, Dict, Any, Optional
import uuid
import time
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import get_document_cleaner
from app.core.embeddings import get_embedding_service
from app.core.chunking import get_smart_chunker
from app.core.logging import get_logger, log_execution_time, metrics_logger
from app.db import async_session
from app.models import LessonChunk, Document
from app.config import settings

logger = get_logger(__name__)


class DocumentProcessor:
    """Process documents through the ingestion pipeline"""
    
    def __init__(self):
        self.cleaner = get_document_cleaner()
        self.chunker = get_smart_chunker()
        self.embedder = get_embedding_service()
        
    @log_execution_time
    async def process_document(
        self,
        text: str,
        document_id: str,
        source_ref: Optional[str] = None,
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Process a single document through the pipeline"""
        
        # 1. Clean document
        logger.info("Cleaning document", document_id=document_id, original_length=len(text))
        start_time = time.time()
        
        cleaned_text = await self.cleaner.clean_document(text, doc_type="web")
        if not cleaned_text or len(cleaned_text.strip()) < 50:
            logger.warning("Document too short after cleaning", 
                         document_id=document_id,
                         cleaned_length=len(cleaned_text) if cleaned_text else 0)
            return []
        
        clean_duration = time.time() - start_time
        logger.info("Document cleaned",
                   document_id=document_id,
                   original_length=len(text),
                   cleaned_length=len(cleaned_text),
                   duration=clean_duration)
        
        # 2. Chunk document
        chunk_metadata = {
            "document_id": document_id,
            "source_ref": source_ref,
            "language": language,
            **(metadata or {})
        }
        
        chunks = await self.chunker.chunk_document(cleaned_text, chunk_metadata)
        metrics_logger.log_chunk_creation(
            count=len(chunks),
            strategy=settings.chunking_strategy.value,
            avg_size=sum(c.metadata.chunk_size for c in chunks) / len(chunks) if chunks else 0
        )
        
        # 3. Generate embeddings
        start_time = time.time()
        chunk_dicts = [chunk.to_dict() for chunk in chunks]
        
        metrics_logger.log_embedding_request(settings.embedding_model, len(chunk_dicts))
        embedded_chunks = await self.embedder.embed_chunks(chunk_dicts)
        
        embed_duration = time.time() - start_time
        metrics_logger.log_embedding_complete(settings.embedding_model, embed_duration)
        
        logger.info("Document processed",
                   document_id=document_id,
                   chunks_created=len(embedded_chunks),
                   total_duration=clean_duration + embed_duration)
        
        return embedded_chunks


class IngestService:
    """High-level service for document ingestion"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        
    @log_execution_time
    async def ingest_text(
        self,
        course_id: str,
        raw_text: str,
        source_ref: Optional[str] = None,
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Ingest text and store chunks in database"""
        
        document_id = str(uuid.uuid4())
        logger.info("Starting text ingestion",
                   course_id=course_id,
                   document_id=document_id,
                   source_ref=source_ref,
                   text_length=len(raw_text))
        
        try:
            # Process document
            chunks = await self.processor.process_document(
                text=raw_text,
                document_id=document_id,
                source_ref=source_ref,
                language=language,
                metadata=metadata
            )
            
            if not chunks:
                logger.warning("No chunks created", document_id=document_id)
                return 0
            
            # Store chunks and document in database
            stored_count = await self._store_chunks(
                course_id=course_id,
                document_id=document_id,
                source_ref=source_ref,
                language=language,
                chunks=chunks
            )
            
            logger.info("Text ingestion completed",
                       course_id=course_id,
                       document_id=document_id,
                       chunks_stored=stored_count)
            
            return stored_count
            
        except Exception as e:
            logger.error("Text ingestion failed",
                        course_id=course_id,
                        document_id=document_id,
                        error=str(e),
                        error_type=type(e).__name__)
            raise
    
    async def _store_chunks(
        self,
        course_id: str,
        document_id: str,
        source_ref: Optional[str],
        language: str,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """Store document and chunks in database"""
        stored_count = 0
        
        async with async_session() as session:
            # Ensure Document exists/created
            try:
                doc_uuid = uuid.UUID(document_id)
            except Exception:
                doc_uuid = uuid.uuid4()

            document = Document(
                id=doc_uuid,
                course_id=course_id,
                title=None,
                source_url=source_ref,
                document_type="web",
                language=language,
                processing_status='processing'
            )
            session.add(document)
            await session.flush()

            for chunk in chunks:
                lesson_chunk = LessonChunk(
                    course_id=course_id,
                    chunk_text=chunk["content"],
                    embedding=chunk.get("embedding"),
                    source_ref=chunk.get("source_ref"),
                    document_id=doc_uuid,
                    embedding_model=chunk.get("embedding_model"),
                    chunk_index=chunk.get("chunk_index"),
                    chunk_size=chunk.get("chunk_size"),
                    search_vector=chunk.get("content"),
                    meta={
                        "chunk_id": chunk.get("chunk_id"),
                        "headers": chunk.get("headers"),
                        "language": chunk.get("language"),
                        "embedding_dimension": chunk.get("embedding_dimension")
                    }
                )
                session.add(lesson_chunk)
                stored_count += 1
            
            # Update document with completion metadata
            document.chunks_count = stored_count
            document.processing_status = 'completed'
            from datetime import datetime
            document.processed_at = datetime.utcnow()
            session.add(document)

            await session.commit()
            
        return stored_count
    
    @log_execution_time
    async def ingest_documents(
        self,
        course_id: str,
        documents: List[Dict[str, Any]],
        language: str = "en"
    ) -> Dict[str, Any]:
        """Ingest multiple documents"""
        
        results = {
            "total_documents": len(documents),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        for doc in documents:
            try:
                chunks_count = await self.ingest_text(
                    course_id=course_id,
                    raw_text=doc.get("content", ""),
                    source_ref=doc.get("source"),
                    language=language,
                    metadata=doc.get("metadata")
                )
                results["successful"] += 1
                results["total_chunks"] += chunks_count
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "document": doc.get("source", "unknown"),
                    "error": str(e)
                })
                logger.error("Document ingestion failed",
                           document=doc.get("source"),
                           error=str(e))
        
        return results
    
    async def ingest_text_content(
        self,
        course_id: str,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Ingest text content with optional title and metadata.
        Used for generated lesson content ingestion.
        """
        return await self.ingest_text(
            course_id=course_id,
            raw_text=content,
            source_ref=title,
            language=metadata.get("language", "en") if metadata else "en",
            metadata=metadata
        )


# Singleton instance
_ingest_service = None


def get_ingest_service() -> IngestService:
    """Get singleton ingest service instance"""
    global _ingest_service
    if _ingest_service is None:
        _ingest_service = IngestService()
    return _ingest_service


# Backward compatibility
async def ingest_text(course_id: str, raw_text: str, source_ref: str | None = None, lang: str = "en"):
    """Legacy function for backward compatibility"""
    service = get_ingest_service()
    return await service.ingest_text(course_id, raw_text, source_ref, lang)