"""
Advanced chunking system with multiple strategies using LangChain
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import uuid
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    LatexTextSplitter,
    HTMLHeaderTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_core.documents import Document
import structlog
import re

from app.config import settings, ChunkingStrategy

logger = structlog.get_logger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for document chunks"""
    chunk_id: str
    document_id: str
    chunk_index: int
    chunk_size: int
    chunk_type: str
    source_ref: Optional[str] = None
    start_position: int = 0
    end_position: int = 0
    headers: Optional[Dict[str, str]] = None
    language: Optional[str] = None
    total_chunks: Optional[int] = None


@dataclass
class DocumentChunk:
    """Document chunk with content and metadata"""
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            "content": self.content,
            "chunk_id": self.metadata.chunk_id,
            "document_id": self.metadata.document_id,
            "chunk_index": self.metadata.chunk_index,
            "chunk_size": self.metadata.chunk_size,
            "chunk_type": self.metadata.chunk_type,
            "source_ref": self.metadata.source_ref,
            "headers": self.metadata.headers,
            "language": self.metadata.language,
            "embedding": self.embedding
        }


class ChunkingStrategyBase(ABC):
    """Base class for chunking strategies"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size or settings.min_chunk_size
        
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text into chunks"""
        pass
    
    def _create_chunks(
        self,
        splits: List[str],
        document_id: str,
        source_ref: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Create DocumentChunk objects from text splits"""
        chunks = []
        position = 0
        
        for i, split in enumerate(splits):
            # Skip empty or too small chunks
            if len(split.strip()) < self.min_chunk_size:
                continue
            
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{i}",
                document_id=document_id,
                chunk_index=i,
                chunk_size=len(split),
                chunk_type=self.__class__.__name__,
                source_ref=source_ref,
                start_position=position,
                end_position=position + len(split),
                headers=additional_metadata.get("headers") if additional_metadata else None,
                language=additional_metadata.get("language") if additional_metadata else None,
                total_chunks=len(splits)
            )
            
            chunk = DocumentChunk(
                content=split.strip(),
                metadata=chunk_metadata
            )
            chunks.append(chunk)
            position += len(split)
        
        return chunks


class FixedChunkingStrategy(ChunkingStrategyBase):
    """Simple fixed-size chunking"""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text into fixed-size chunks"""
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n",
            length_function=len
        )
        
        splits = splitter.split_text(text)
        logger.info("Fixed chunking completed", 
                   chunks_created=len(splits),
                   document_id=metadata.get("document_id"))
        
        return self._create_chunks(
            splits,
            metadata.get("document_id", str(uuid.uuid4())),
            metadata.get("source_ref"),
            metadata
        )


class RecursiveChunkingStrategy(ChunkingStrategyBase):
    """Recursive chunking with multiple separators"""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text recursively with fallback separators"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
            keep_separator=True
        )
        
        splits = splitter.split_text(text)
        logger.info("Recursive chunking completed",
                   chunks_created=len(splits),
                   document_id=metadata.get("document_id"))
        
        return self._create_chunks(
            splits,
            metadata.get("document_id", str(uuid.uuid4())),
            metadata.get("source_ref"),
            metadata
        )


class SemanticChunkingStrategy(ChunkingStrategyBase):
    """Semantic chunking based on sentence boundaries"""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text semantically at sentence boundaries"""
        # Use sentence transformer token splitter for semantic boundaries
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        splits = splitter.split_text(text)
        logger.info("Semantic chunking completed",
                   chunks_created=len(splits),
                   document_id=metadata.get("document_id"))
        
        return self._create_chunks(
            splits,
            metadata.get("document_id", str(uuid.uuid4())),
            metadata.get("source_ref"),
            metadata
        )


class MarkdownChunkingStrategy(ChunkingStrategyBase):
    """Specialized chunking for Markdown documents"""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split Markdown preserving structure"""
        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Split into documents
        documents = splitter.create_documents([text])
        
        # Extract headers and create chunks
        chunks = []
        document_id = metadata.get("document_id", str(uuid.uuid4()))
        
        for i, doc in enumerate(documents):
            # Extract headers from content
            headers = self._extract_markdown_headers(doc.page_content)
            
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{i}",
                document_id=document_id,
                chunk_index=i,
                chunk_size=len(doc.page_content),
                chunk_type=self.__class__.__name__,
                source_ref=metadata.get("source_ref"),
                headers=headers,
                language=metadata.get("language"),
                total_chunks=len(documents)
            )
            
            chunk = DocumentChunk(
                content=doc.page_content.strip(),
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        
        logger.info("Markdown chunking completed",
                   chunks_created=len(chunks),
                   document_id=document_id)
        
        return chunks
    
    def _extract_markdown_headers(self, content: str) -> Dict[str, str]:
        """Extract headers from markdown content"""
        headers = {}
        
        # Extract different header levels
        h1_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        h2_match = re.search(r'^## (.+)$', content, re.MULTILINE)
        h3_match = re.search(r'^### (.+)$', content, re.MULTILINE)
        
        if h1_match:
            headers['h1'] = h1_match.group(1).strip()
        if h2_match:
            headers['h2'] = h2_match.group(1).strip()
        if h3_match:
            headers['h3'] = h3_match.group(1).strip()
        
        return headers


class ChunkingFactory:
    """Factory for creating chunking strategies"""
    
    _strategies = {
        ChunkingStrategy.FIXED: FixedChunkingStrategy,
        ChunkingStrategy.RECURSIVE: RecursiveChunkingStrategy,
        ChunkingStrategy.SEMANTIC: SemanticChunkingStrategy,
        ChunkingStrategy.MARKDOWN: MarkdownChunkingStrategy,
    }
    
    @classmethod
    def create(
        cls,
        strategy: ChunkingStrategy = None,
        **kwargs
    ) -> ChunkingStrategyBase:
        """Create a chunking strategy instance"""
        strategy = strategy or settings.chunking_strategy
        
        if strategy not in cls._strategies:
            logger.warning(f"Unknown strategy {strategy}, falling back to recursive")
            strategy = ChunkingStrategy.RECURSIVE
        
        strategy_class = cls._strategies[strategy]
        return strategy_class(**kwargs)


class SmartChunker:
    """Intelligent chunker that selects strategy based on content"""
    
    def __init__(self):
        self.factory = ChunkingFactory()
        
    def detect_content_type(self, text: str) -> ChunkingStrategy:
        """Detect the best chunking strategy for content"""
        # Check for markdown indicators
        markdown_patterns = [r'^#{1,6}\s', r'^\*{1,3}\s', r'^\-\s', r'```', r'\[.*\]\(.*\)']
        markdown_score = sum(1 for pattern in markdown_patterns 
                           if re.search(pattern, text[:1000], re.MULTILINE))
        
        if markdown_score >= 2:
            return ChunkingStrategy.MARKDOWN
        
        # Check for code patterns
        code_patterns = [r'def\s+\w+', r'class\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import']
        code_score = sum(1 for pattern in code_patterns 
                        if re.search(pattern, text[:1000]))
        
        if code_score >= 2:
            return ChunkingStrategy.SEMANTIC
        
        # Default to recursive
        return ChunkingStrategy.RECURSIVE
    
    async def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        strategy: Optional[ChunkingStrategy] = None
    ) -> List[DocumentChunk]:
        """Chunk document with auto-detection or specified strategy"""
        if not text or not text.strip():
            return []
        
        # Auto-detect strategy if not specified
        if strategy is None:
            strategy = self.detect_content_type(text)
            logger.info("Auto-detected chunking strategy",
                       strategy=strategy.value,
                       document_id=metadata.get("document_id"))
        
        # Create chunker and process
        chunker = self.factory.create(strategy)
        chunks = chunker.chunk(text, metadata)
        
        logger.info("Document chunked",
                   strategy=strategy.value,
                   chunks_created=len(chunks),
                   document_id=metadata.get("document_id"),
                   avg_chunk_size=sum(c.metadata.chunk_size for c in chunks) / len(chunks) if chunks else 0)
        
        return chunks
    
    async def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "content",
        strategy: Optional[ChunkingStrategy] = None
    ) -> List[DocumentChunk]:
        """Chunk multiple documents"""
        all_chunks = []
        
        for doc in documents:
            text = doc.get(text_field, "")
            if not text:
                continue
            
            # Prepare metadata
            metadata = {
                "document_id": doc.get("id", str(uuid.uuid4())),
                "source_ref": doc.get("source"),
                "language": doc.get("language", "en"),
                **{k: v for k, v in doc.items() if k not in ["content", "id", "source"]}
            }
            
            chunks = await self.chunk_document(text, metadata, strategy)
            all_chunks.extend(chunks)
        
        return all_chunks


# Singleton instance
_smart_chunker = None


def get_smart_chunker() -> SmartChunker:
    """Get singleton smart chunker instance"""
    global _smart_chunker
    if _smart_chunker is None:
        _smart_chunker = SmartChunker()
    return _smart_chunker
