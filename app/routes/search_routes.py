"""
Search routes for RAG functionality
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

from app.core.vector_search import get_search_engine
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ai/search", tags=["search"])


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., description="Search query text")
    course_id: Optional[str] = Field(None, description="Filter by course ID")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    use_hybrid: bool = Field(True, description="Use hybrid search (vector + keyword)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional metadata filters")


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_type: str
    filters_applied: Optional[Dict[str, Any]]


@router.post("/", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform vector/hybrid search on ingested documents
    """
    try:
        search_engine = get_search_engine()
        
        results = await search_engine.search(
            query=request.query,
            course_id=request.course_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            filters=request.filters,
            use_hybrid=request.use_hybrid
        )
        
        return SearchResponse(
            query=request.query,
            results=[result.to_dict() for result in results],
            total_results=len(results),
            search_type="hybrid" if request.use_hybrid else "vector",
            filters_applied=request.filters
        )
        
    except Exception as e:
        logger.error("Search failed", error=str(e), query=request.query)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/similar/{chunk_id}")
async def find_similar_chunks(
    chunk_id: int,
    top_k: int = Query(5, ge=1, le=20),
    similarity_threshold: float = Query(0.8, ge=0.0, le=1.0)
):
    """
    Find chunks similar to a given chunk
    """
    try:
        search_engine = get_search_engine()
        
        results = await search_engine.get_similar_chunks(
            chunk_id=chunk_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        return {
            "source_chunk_id": chunk_id,
            "similar_chunks": [result.to_dict() for result in results],
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error("Similar chunks search failed", error=str(e), chunk_id=chunk_id)
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")


@router.post("/metadata")
async def search_by_metadata(metadata_filters: Dict[str, Any]):
    """
    Search chunks by metadata only
    """
    try:
        search_engine = get_search_engine()
        
        results = await search_engine.search_by_metadata(
            metadata_filters=metadata_filters
        )
        
        return {
            "filters": metadata_filters,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error("Metadata search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metadata search failed: {str(e)}")
