"""
Modern embeddings module using LangChain with support for multiple providers
"""
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
import asyncio

from app.config import settings

logger = structlog.get_logger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass


class LangChainEmbeddingProvider(EmbeddingProvider):
    """LangChain-based embedding provider with multi-model support"""
    
    def __init__(self):
        self.embeddings = self._initialize_embeddings()
        self.dimension = self._get_dimension()
        
    def _initialize_embeddings(self) -> Embeddings:
        """Initialize the appropriate embeddings based on configuration"""
        if "embedding-001" in settings.embedding_model or "gemini" in settings.embedding_model:
            return GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model,
                google_api_key=settings.gemini_api_key,
                task_type="retrieval_document"
            )
        elif settings.openai_api_key and "text-embedding" in settings.embedding_model:
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
                dimensions=settings.embedding_dim
            )
        else:
            # Default to local HuggingFace model
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def _get_dimension(self) -> int:
        """Get embedding dimension for the model"""
        model_dimensions = {
            "models/embedding-001": 768,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768
        }
        return model_dimensions.get(settings.embedding_model, settings.embedding_dim)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with retry logic"""
        if not texts:
            return []
        
        try:
            # Clean and validate texts
            cleaned_texts = []
            for text in texts:
                if text and text.strip():
                    # Truncate if too long
                    if len(text) > settings.max_embedding_length:
                        text = text[:settings.max_embedding_length]
                    cleaned_texts.append(text)
            
            if not cleaned_texts:
                return []
            
            # Process in batches to avoid rate limits
            embeddings = []
            batch_size = settings.embedding_batch_size
            
            for i in range(0, len(cleaned_texts), batch_size):
                batch = cleaned_texts[i:i + batch_size]
                
                # Use asyncio.to_thread for sync embeddings
                batch_embeddings = await asyncio.to_thread(
                    self.embeddings.embed_documents,
                    batch
                )
                embeddings.extend(batch_embeddings)
                
                # Rate limiting between batches
                if i + batch_size < len(cleaned_texts):
                    await asyncio.sleep(0.1)
            
            logger.info("Documents embedded",
                       count=len(texts),
                       model=settings.embedding_model,
                       dimension=self.dimension)
            
            return embeddings
            
        except Exception as e:
            logger.error("Embedding generation failed",
                        error=str(e),
                        model=settings.embedding_model,
                        text_count=len(texts))
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if not text or not text.strip():
            return []
        
        try:
            # Truncate if too long
            if len(text) > settings.max_embedding_length:
                text = text[:settings.max_embedding_length]
            
            # For Google embeddings, we need to handle task_type
            if isinstance(self.embeddings, GoogleGenerativeAIEmbeddings):
                # Create a new instance with query task type
                query_embeddings = GoogleGenerativeAIEmbeddings(
                    model=settings.embedding_model,
                    google_api_key=settings.gemini_api_key,
                    task_type="retrieval_query"
                )
                embedding = await asyncio.to_thread(
                    query_embeddings.embed_query,
                    text
                )
            else:
                embedding = await asyncio.to_thread(
                    self.embeddings.embed_query,
                    text
                )
            
            logger.debug("Query embedded",
                        model=settings.embedding_model,
                        text_length=len(text))
            
            return embedding
            
        except Exception as e:
            logger.error("Query embedding failed",
                        error=str(e),
                        model=settings.embedding_model)
            raise


class EmbeddingService:
    """High-level service for embedding operations"""
    
    def __init__(self, provider: Optional[EmbeddingProvider] = None):
        self.provider = provider or LangChainEmbeddingProvider()
        
    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query (proxy to provider)."""
        return await self.provider.embed_query(query)
        
    async def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed a list of document chunks"""
        # Extract texts
        texts = [chunk.get("content", "") for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.provider.embed_documents(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
            chunk["embedding_model"] = settings.embedding_model
            chunk["embedding_dimension"] = len(embedding) if embedding else 0
        
        return chunks
    
    async def embed_query_with_expansion(self, query: str, expansion_terms: List[str] = None) -> Dict[str, Any]:
        """Embed query with optional expansion terms"""
        # Embed main query
        query_embedding = await self.provider.embed_query(query)
        
        result = {
            "query": query,
            "embedding": query_embedding,
            "model": settings.embedding_model
        }
        
        # Embed expansion terms if provided
        if expansion_terms:
            expansion_embeddings = await self.provider.embed_documents(expansion_terms)
            result["expansions"] = [
                {"term": term, "embedding": emb}
                for term, emb in zip(expansion_terms, expansion_embeddings)
            ]
        
        return result
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        if not embedding1 or not embedding2:
            return 0.0
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    async def find_similar(
        self,
        query_embedding: List[float],
        document_embeddings: List[Dict[str, Any]],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find similar documents based on embeddings"""
        similarities = []
        
        for doc in document_embeddings:
            if "embedding" in doc and doc["embedding"]:
                similarity = self.compute_similarity(query_embedding, doc["embedding"])
                if similarity >= threshold:
                    similarities.append({
                        **doc,
                        "similarity": similarity
                    })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:top_k]


# Singleton instances
_embedding_provider = None
_embedding_service = None


def get_embedding_provider() -> EmbeddingProvider:
    """Get singleton embedding provider instance"""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = LangChainEmbeddingProvider()
    return _embedding_provider


def get_embedding_service() -> EmbeddingService:
    """Get singleton embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
