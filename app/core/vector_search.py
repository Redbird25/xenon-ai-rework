"""
Advanced vector search with hybrid search capabilities
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import numpy as np
from sqlalchemy import text, and_, or_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.db import async_session
from app.models import LessonChunk, Document, SearchQuery
from app.core.embeddings import get_embedding_service
from app.core.llm import get_query_processor
from app.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """Search result with relevance score"""
    chunk_id: int
    content: str
    source_ref: Optional[str]
    metadata: Dict[str, Any]
    similarity_score: float
    text_score: float
    combined_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_ref": self.source_ref,
            "metadata": self.metadata,
            "scores": {
                "similarity": self.similarity_score,
                "text": self.text_score,
                "combined": self.combined_score
            }
        }


class VectorSearchEngine:
    """Advanced vector search engine with hybrid capabilities"""
    
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.query_processor = get_query_processor()
        
    async def search(
        self,
        query: str,
        course_id: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
        track_query: bool = True
    ) -> List[SearchResult]:
        """Perform vector search with optional hybrid search"""
        
        start_time = time.time()
        
        # Enhance query
        enhanced_query_data = await self.query_processor.enhance_query(query)
        enhanced_query = enhanced_query_data.get("enhanced_query", query)
        keywords = enhanced_query_data.get("keywords", [])
        
        # Generate query embedding from ORIGINAL query to avoid semantic drift
        query_embedding = await self.embedding_service.embed_query(query)
        
        # Perform search
        if use_hybrid:
            results = await self._hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                keywords=keywords,
                course_id=course_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=filters
            )
            # Fallback: relax threshold if no results
            if not results and similarity_threshold > 0.5:
                results = await self._hybrid_search(
                    query_embedding=query_embedding,
                    query_text=query,
                    keywords=keywords,
                    course_id=course_id,
                    top_k=top_k,
                    similarity_threshold=0.5,
                    filters=filters
                )
        else:
            results = await self._vector_search(
                query_embedding=query_embedding,
                course_id=course_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=filters
            )
            # Fallback: relax threshold if no results
            if not results and similarity_threshold > 0.5:
                results = await self._vector_search(
                    query_embedding=query_embedding,
                    course_id=course_id,
                    top_k=top_k,
                    similarity_threshold=0.5,
                    filters=filters
                )
        
        # Track query for analytics
        if track_query:
            await self._track_search_query(
                query=query,
                enhanced_query=enhanced_query,
                query_embedding=query_embedding,
                course_id=course_id,
                results=results,
                response_time_ms=int((time.time() - start_time) * 1000),
                filters=filters
            )
        
        logger.info("Search completed",
                   query=query,
                   results_count=len(results),
                   response_time_ms=int((time.time() - start_time) * 1000),
                   use_hybrid=use_hybrid)
        
        return results
    
    async def _vector_search(
        self,
        query_embedding: List[float],
        course_id: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Pure vector similarity search"""
        
        async with async_session() as session:
            # Build parameterized SQL using pgvector cosine operator <=>
            embedding_literal = '[' + ','.join(str(x) for x in query_embedding) + ']'

            sql_parts = [
                "SELECT id, chunk_text, source_ref, metadata, "
                "1 - (embedding <=> CAST(:embedding AS vector)) AS similarity "
                "FROM lesson_chunks WHERE embedding IS NOT NULL"
            ]

            params: Dict[str, Any] = {
                'embedding': embedding_literal,
                'threshold': similarity_threshold,
                'limit': top_k
            }

            if course_id:
                sql_parts.append("AND course_id = :course_id")
                params['course_id'] = str(course_id)

            # Apply metadata/date filters if provided
            if filters:
                if 'language' in filters:
                    sql_parts.append("AND (metadata->>'language') = :language")
                    params['language'] = str(filters['language'])
                if 'document_type' in filters:
                    sql_parts.append("AND (metadata->>'document_type') = :document_type")
                    params['document_type'] = str(filters['document_type'])
                if 'date_from' in filters:
                    sql_parts.append("AND created_at >= :date_from")
                    params['date_from'] = filters['date_from']
                if 'date_to' in filters:
                    sql_parts.append("AND created_at <= :date_to")
                    params['date_to'] = filters['date_to']

            # Similarity threshold, ordering, and limiting
            sql_parts.append("AND (1 - (embedding <=> CAST(:embedding AS vector))) > :threshold")
            sql_parts.append("ORDER BY similarity DESC")
            sql_parts.append("LIMIT :limit")

            full_sql = ' '.join(sql_parts)

            result = await session.execute(text(full_sql), params)
            rows = result.fetchall()

            search_results: List[SearchResult] = []
            for row in rows:
                search_results.append(SearchResult(
                    chunk_id=row.id,
                    content=row.chunk_text,
                    source_ref=row.source_ref,
                    metadata=row.metadata or {},
                    similarity_score=float(row.similarity),
                    text_score=0.0,
                    combined_score=float(row.similarity)
                ))

            return search_results
    
    async def _hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        keywords: List[str],
        course_id: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5  # Weight for vector vs text search
    ) -> List[SearchResult]:
        """Hybrid search combining vector and keyword search"""
        
        async with async_session() as session:
            # Use the hybrid_search database function
            query_params = {
                'query_embedding': str(query_embedding),
                'query_text': query_text,
                'course_id_filter': course_id,
                'similarity_threshold': similarity_threshold,
                'limit_results': top_k * 2  # Get more results for reranking
            }
            
            # Execute hybrid search
            result = await session.execute(
                text(
                    "SELECT * FROM hybrid_search("
                    "CAST(:query_embedding AS vector), "
                    ":query_text, :course_id_filter, :similarity_threshold, :limit_results)"
                ),
                query_params
            )
            
            rows = result.fetchall()
            
            # Prefetch metadata for all rows to avoid N+1
            id_list = [r.id for r in rows]
            meta_map: Dict[int, Dict[str, Any]] = {}
            if id_list:
                stmt = select(LessonChunk.id, LessonChunk.meta)
                stmt = stmt.where(LessonChunk.id.in_(id_list))
                meta_rows = await session.execute(stmt)
                for mid, mval in meta_rows.all():
                    meta_map[int(mid)] = mval or {}

            # Calculate combined scores and build results
            search_results: List[SearchResult] = []
            for row in rows:
                vector_score = row.similarity
                text_score = row.rank

                # Normalize text score to 0-1 range
                if text_score and text_score > 0:
                    text_score = min(text_score / 5.0, 1.0)  # Assuming max rank ~5
                else:
                    text_score = 0.0

                combined_score = alpha * vector_score + (1 - alpha) * text_score

                search_results.append(SearchResult(
                    chunk_id=row.id,
                    content=row.chunk_text,
                    source_ref=row.source_ref,
                    metadata=meta_map.get(int(row.id), {}),
                    similarity_score=float(vector_score),
                    text_score=float(text_score),
                    combined_score=float(combined_score)
                ))
            
            # Sort by combined score and limit
            search_results.sort(key=lambda x: x.combined_score, reverse=True)
            return search_results[:top_k]
    
    def _apply_filters(self, query, filters: Dict[str, Any]):
        """Apply metadata filters to query"""
        if 'language' in filters:
            query = query.where(
                LessonChunk.meta['language'].astext == str(filters['language'])
            )

        if 'document_type' in filters:
            query = query.where(
                LessonChunk.meta['document_type'].astext == str(filters['document_type'])
            )

        if 'date_from' in filters:
            query = query.where(
                LessonChunk.created_at >= filters['date_from']
            )

        if 'date_to' in filters:
            query = query.where(
                LessonChunk.created_at <= filters['date_to']
            )

        return query
    
    async def _track_search_query(
        self,
        query: str,
        enhanced_query: str,
        query_embedding: List[float],
        course_id: Optional[str],
        results: List[SearchResult],
        response_time_ms: int,
        filters: Optional[Dict[str, Any]]
    ):
        """Track search query for analytics"""
        try:
            async with async_session() as session:
                search_query = SearchQuery(
                    course_id=course_id,
                    query_text=query,
                    enhanced_query=enhanced_query,
                    query_embedding=query_embedding,
                    search_params={
                        "top_k": len(results),
                        "similarity_threshold": settings.similarity_threshold
                    },
                    filters=filters or {},
                    results_count=len(results),
                    top_score=results[0].combined_score if results else 0.0,
                    response_time_ms=response_time_ms
                )
                session.add(search_query)
                await session.commit()
        except Exception as e:
            logger.error("Failed to track search query", error=str(e))
    
    async def get_similar_chunks(
        self,
        chunk_id: int,
        top_k: int = 5,
        similarity_threshold: float = 0.8
    ) -> List[SearchResult]:
        """Find chunks similar to a given chunk"""
        
        async with async_session() as session:
            # Get the source chunk
            source_chunk = await session.get(LessonChunk, chunk_id)
            if not source_chunk or not source_chunk.embedding:
                return []
            
            # Search for similar chunks
            results = await self._vector_search(
                query_embedding=source_chunk.embedding,
                course_id=str(source_chunk.course_id) if source_chunk.course_id else None,
                top_k=top_k + 1,  # +1 to exclude self
                similarity_threshold=similarity_threshold
            )

            # Exclude the source chunk itself and return top_k
            filtered = [r for r in results if r.chunk_id != source_chunk.id]
            return filtered[:top_k]
    
    async def search_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        course_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search chunks by metadata only"""
        
        async with async_session() as session:
            stmt = select(LessonChunk)

            if course_id:
                stmt = stmt.where(LessonChunk.course_id == course_id)

            # Apply metadata filters
            for key, value in metadata_filters.items():
                stmt = stmt.where(LessonChunk.meta[key].astext == str(value))

            stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            chunks = result.scalars().all()

            return [chunk.to_dict() for chunk in chunks]


# Singleton instance
_search_engine = None


def get_search_engine() -> VectorSearchEngine:
    """Get singleton search engine instance"""
    global _search_engine
    if _search_engine is None:
        _search_engine = VectorSearchEngine()
    return _search_engine
