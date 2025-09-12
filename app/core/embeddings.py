"""
Modern embeddings module using LangChain with support for multiple providers
"""
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import os
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import structlog
import asyncio
import time
from datetime import datetime, timedelta

from app.config import settings
import tiktoken

logger = structlog.get_logger(__name__)

# Global rate limiter for Gemini embeddings
class GeminiRateLimiter:
    def __init__(self):
        self.requests_per_minute = settings.gemini_rpm_limit
        self.requests_per_day = settings.gemini_rpd_limit
        self.tokens_per_minute = 30000  # Gemini Free tier TPM limit
        self.request_times = []        # Store request timestamps
        self.token_usage = []          # Store (timestamp, token_count) pairs
        self.daily_requests = 0
        self.day_start = datetime.now().date()
    
    async def wait_if_needed(self, estimated_tokens: int = 0):
        """Wait if rate limit would be exceeded"""
        now = datetime.now()
        
        # Reset daily counter if new day
        if now.date() > self.day_start:
            self.daily_requests = 0
            self.day_start = now.date()
        
        # Check daily limit
        if self.daily_requests >= self.requests_per_day:
            raise Exception(f"Daily rate limit exceeded ({self.requests_per_day} requests per day)")
        
        # Clean old requests (older than 1 minute)
        minute_ago = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > minute_ago]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > minute_ago]
        
        # Check RPM limit
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - min(self.request_times)).total_seconds()
            if sleep_time > 0:
                logger.info(f"RPM limit: sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        # Check TPM limit
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        if current_tokens + estimated_tokens > self.tokens_per_minute:
            # Calculate wait time based on oldest token usage
            if self.token_usage:
                oldest_time = min(t for t, _ in self.token_usage)
                sleep_time = 60 - (now - oldest_time).total_seconds()
                if sleep_time > 0:
                    logger.info(f"TPM limit: sleeping {sleep_time:.1f}s (tokens: {current_tokens + estimated_tokens})")
                    await asyncio.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(now)
        self.token_usage.append((now, estimated_tokens))
        self.daily_requests += 1

gemini_rate_limiter = GeminiRateLimiter()


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    try:
        # Use cl100k_base encoder (GPT-4, GPT-3.5-turbo)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


def create_token_aware_batches(texts: List[str], max_tokens_per_batch: int = 2000) -> List[List[str]]:
    """Create batches of texts that stay under token limit"""
    batches = []
    current_batch = []
    current_tokens = 0
    
    for text in texts:
        text_tokens = count_tokens(text)
        
        # If single text exceeds limit, truncate it
        if text_tokens > max_tokens_per_batch:
            # Truncate text to fit in batch
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            truncated_tokens = tokens[:max_tokens_per_batch]
            text = encoding.decode(truncated_tokens)
            text_tokens = max_tokens_per_batch
        
        # If adding this text would exceed batch limit, start new batch
        if current_batch and current_tokens + text_tokens > max_tokens_per_batch:
            batches.append(current_batch)
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens
    
    # Add final batch if not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches


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
            # Default to local HuggingFace model - high quality
            # Set up cache directory to avoid permission issues
            cache_dir = Path("./hf_cache")
            cache_dir.mkdir(exist_ok=True)
            
            # Set environment variables for HuggingFace cache
            os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)
            os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)
            
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                cache_folder=str(cache_dir)
            )
    
    def _get_dimension(self) -> int:
        """Get embedding dimension for the model"""
        model_dimensions = {
            "models/embedding-001": 768,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "sentence-transformers/all-mpnet-base-v2": 768
        }
        return model_dimensions.get(settings.embedding_model, settings.embedding_dim)
    
    def _create_fallback_embeddings(self) -> HuggingFaceEmbeddings:
        """Create fallback local embeddings"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with retry logic"""
        if not texts:
            return []
        
        try:
            return await self._embed_with_fallback(texts, is_query=False)
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
            result = await self._embed_with_fallback([text], is_query=True)
            return result[0] if result else []
        except Exception as e:
            logger.error("Query embedding failed",
                        error=str(e),
                        model=settings.embedding_model)
            raise
    
    async def _embed_with_fallback(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Embed with fallback to local model on rate limit"""
        if not texts:
            return []
        
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
        
        # Try primary model first
        try:
            return await self._embed_with_primary_model(cleaned_texts, is_query)
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit or quota error
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                logger.warning("Primary embedding model rate limited, falling back to local model",
                              model=settings.embedding_model,
                              error=error_str)
                return await self._embed_with_fallback_model(cleaned_texts)
            else:
                # For other errors, still try fallback but log as error
                logger.error("Primary embedding model failed, trying fallback",
                           model=settings.embedding_model,
                           error=error_str)
                try:
                    return await self._embed_with_fallback_model(cleaned_texts)
                except Exception as fallback_error:
                    logger.error("Fallback embedding also failed",
                               fallback_error=str(fallback_error))
                    raise e  # Raise original error
    
    async def _embed_with_primary_model(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Embed using primary model with token-aware batching and rate limiting"""
        embeddings = []
        
        # Handle Gemini with special token-aware batching
        if isinstance(self.embeddings, GoogleGenerativeAIEmbeddings):
            # Create token-aware batches for Gemini (2048 token limit)
            token_batches = create_token_aware_batches(texts, max_tokens_per_batch=2000)
            
            logger.info("Token-aware batching for Gemini",
                       total_texts=len(texts),
                       total_batches=len(token_batches),
                       model=settings.embedding_model)
            
            for batch_idx, batch in enumerate(token_batches):
                # Calculate total tokens for this batch
                batch_tokens = sum(count_tokens(text) for text in batch)
                
                # Apply rate limiting with token estimation
                await gemini_rate_limiter.wait_if_needed(estimated_tokens=batch_tokens)
                
                if is_query and len(batch) == 1:
                    # Handle single query embedding with proper task type
                    query_embeddings = GoogleGenerativeAIEmbeddings(
                        model=settings.embedding_model,
                        google_api_key=settings.gemini_api_key,
                        task_type="retrieval_query"
                    )
                    embedding = await asyncio.to_thread(
                        query_embeddings.embed_query,
                        batch[0]
                    )
                    embeddings.append(embedding)
                else:
                    # Handle batch document embedding
                    batch_embeddings = await asyncio.to_thread(
                        self.embeddings.embed_documents,
                        batch
                    )
                    embeddings.extend(batch_embeddings)
                
                logger.info("Gemini batch processed",
                           batch=f"{batch_idx+1}/{len(token_batches)}",
                           batch_size=len(batch),
                           batch_tokens=batch_tokens)
                
                # Rate limiting between batches - longer delay for Gemini
                if batch_idx + 1 < len(token_batches):
                    await asyncio.sleep(2.0)  # 2 second delay between token batches
        
        else:
            # Standard batching for other providers
            batch_size = min(settings.embedding_batch_size, 32)  # Larger batches for non-Gemini
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                if is_query and len(batch) == 1:
                    embedding = await asyncio.to_thread(
                        self.embeddings.embed_query,
                        batch[0]
                    )
                    embeddings.append(embedding)
                else:
                    # Handle batch document embedding
                    batch_embeddings = await asyncio.to_thread(
                        self.embeddings.embed_documents,
                        batch
                    )
                    embeddings.extend(batch_embeddings)
                
                # Rate limiting between batches
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.5)  # Shorter delay for other providers
        
        logger.info("Documents embedded with primary model",
                   count=len(texts),
                   model=settings.embedding_model,
                   dimension=self.dimension)
        
        return embeddings
    
    async def _embed_with_fallback_model(self, texts: List[str]) -> List[List[float]]:
        """Embed using local fallback model"""
        fallback_embeddings = self._create_fallback_embeddings()
        
        # Process in batches
        embeddings = []
        batch_size = 32  # Larger batches for local model
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await asyncio.to_thread(
                fallback_embeddings.embed_documents,
                batch
            )
            embeddings.extend(batch_embeddings)
        
        logger.info("Documents embedded with fallback model",
                   count=len(texts),
                   model="sentence-transformers/all-mpnet-base-v2")
        
        return embeddings


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
