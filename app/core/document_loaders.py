"""
Document loaders for various file formats
"""
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import asyncio
import hashlib
from pathlib import Path
import aiohttp
from bs4 import BeautifulSoup
import pypdf
import markdown
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
    JSONLoader
)
from langchain_core.documents import Document as LangChainDocument
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class DocumentLoader(ABC):
    """Base class for document loaders"""
    
    @abstractmethod
    async def load(self, source: str) -> Dict[str, Any]:
        """Load document from source"""
        pass
    
    @abstractmethod
    def can_handle(self, source: str) -> bool:
        """Check if loader can handle this source"""
        pass
    
    def calculate_hash(self, content: str) -> str:
        """Calculate content hash for deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()


class WebDocumentLoader(DocumentLoader):
    """Load documents from web URLs"""
    
    def can_handle(self, source: str) -> bool:
        return source.startswith(('http://', 'https://'))
    
    async def load(self, source: str) -> Dict[str, Any]:
        """Load web document"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(source, timeout=30) as response:
                    response.raise_for_status()
                    content = await response.text()
                    content_type = response.headers.get('content-type', '')
                    
                    # Parse HTML if needed
                    if 'text/html' in content_type:
                        soup = BeautifulSoup(content, 'lxml')
                        
                        # Extract metadata
                        title = soup.find('title')
                        title = title.text if title else None
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Get text
                        text = soup.get_text(separator='\n', strip=True)
                        
                        # Extract other metadata
                        meta_description = soup.find('meta', attrs={'name': 'description'})
                        description = meta_description.get('content') if meta_description else None
                        
                        return {
                            'content': text,
                            'source': source,
                            'title': title,
                            'document_type': 'html',
                            'metadata': {
                                'description': description,
                                'content_type': content_type
                            },
                            'content_hash': self.calculate_hash(text)
                        }
                    else:
                        # Plain text or other formats
                        return {
                            'content': content,
                            'source': source,
                            'document_type': 'text',
                            'metadata': {
                                'content_type': content_type
                            },
                            'content_hash': self.calculate_hash(content)
                        }
                        
        except Exception as e:
            logger.error("Failed to load web document", source=source, error=str(e))
            raise


class PDFDocumentLoader(DocumentLoader):
    """Load PDF documents"""
    
    def can_handle(self, source: str) -> bool:
        return source.endswith('.pdf') or 'application/pdf' in source
    
    async def load(self, source: str) -> Dict[str, Any]:
        """Load PDF document"""
        try:
            # Download if URL
            if source.startswith(('http://', 'https://')):
                content = await self._download_file(source)
                temp_path = Path(f"/tmp/{hashlib.md5(source.encode()).hexdigest()}.pdf")
                temp_path.write_bytes(content)
                source = str(temp_path)
            
            # Use LangChain PDF loader
            loader = PyPDFLoader(source)
            documents = await asyncio.to_thread(loader.load)
            
            # Combine pages
            full_text = '\n\n'.join([doc.page_content for doc in documents])
            
            # Extract metadata
            metadata = {
                'page_count': len(documents),
                'source_type': 'pdf'
            }
            
            if documents and documents[0].metadata:
                metadata.update(documents[0].metadata)
            
            return {
                'content': full_text,
                'source': source,
                'document_type': 'pdf',
                'metadata': metadata,
                'content_hash': self.calculate_hash(full_text)
            }
            
        except Exception as e:
            logger.error("Failed to load PDF document", source=source, error=str(e))
            raise
    
    async def _download_file(self, url: str) -> bytes:
        """Download file from URL"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.read()


class MarkdownDocumentLoader(DocumentLoader):
    """Load Markdown documents"""
    
    def can_handle(self, source: str) -> bool:
        return source.endswith('.md') or source.endswith('.markdown')
    
    async def load(self, source: str) -> Dict[str, Any]:
        """Load Markdown document"""
        try:
            if source.startswith(('http://', 'https://')):
                # Download from URL
                async with aiohttp.ClientSession() as session:
                    async with session.get(source) as response:
                        response.raise_for_status()
                        content = await response.text()
            else:
                # Load from file
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Parse markdown to extract structure
            md = markdown.Markdown(extensions=['meta', 'toc', 'tables', 'fenced_code'])
            html = md.convert(content)
            
            # Extract metadata
            metadata = {
                'format': 'markdown',
                'toc': getattr(md, 'toc', None),
                'meta': getattr(md, 'Meta', {})
            }
            
            return {
                'content': content,  # Keep original markdown
                'source': source,
                'document_type': 'markdown',
                'metadata': metadata,
                'content_hash': self.calculate_hash(content)
            }
            
        except Exception as e:
            logger.error("Failed to load Markdown document", source=source, error=str(e))
            raise


class TextDocumentLoader(DocumentLoader):
    """Load plain text documents"""
    
    def can_handle(self, source: str) -> bool:
        text_extensions = ['.txt', '.text', '.log', '.csv', '.json']
        return any(source.endswith(ext) for ext in text_extensions)
    
    async def load(self, source: str) -> Dict[str, Any]:
        """Load text document"""
        try:
            if source.startswith(('http://', 'https://')):
                # Download from URL
                async with aiohttp.ClientSession() as session:
                    async with session.get(source) as response:
                        response.raise_for_status()
                        content = await response.text()
            else:
                # Load from file
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Detect specific formats
            document_type = 'text'
            if source.endswith('.csv'):
                document_type = 'csv'
            elif source.endswith('.json'):
                document_type = 'json'
            
            return {
                'content': content,
                'source': source,
                'document_type': document_type,
                'metadata': {
                    'encoding': 'utf-8'
                },
                'content_hash': self.calculate_hash(content)
            }
            
        except Exception as e:
            logger.error("Failed to load text document", source=source, error=str(e))
            raise


class DocumentLoaderFactory:
    """Factory for creating appropriate document loaders"""
    
    def __init__(self):
        self.loaders = [
            WebDocumentLoader(),
            PDFDocumentLoader(),
            MarkdownDocumentLoader(),
            TextDocumentLoader()
        ]
    
    async def load_document(self, source: str) -> Dict[str, Any]:
        """Load document using appropriate loader"""
        for loader in self.loaders:
            if loader.can_handle(source):
                logger.info("Loading document", 
                          source=source, 
                          loader=loader.__class__.__name__)
                return await loader.load(source)
        
        # Default to text loader
        logger.warning("No specific loader found, using text loader", source=source)
        return await TextDocumentLoader().load(source)
    
    async def load_documents(self, sources: List[str]) -> List[Dict[str, Any]]:
        """Load multiple documents concurrently"""
        tasks = [self.load_document(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        documents = []
        errors = []
        
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                errors.append({
                    'source': source,
                    'error': str(result)
                })
                logger.error("Document loading failed", 
                           source=source, 
                           error=str(result))
            else:
                documents.append(result)
        
        if errors:
            logger.warning("Some documents failed to load", 
                         total=len(sources), 
                         failed=len(errors))
        
        return documents


# Singleton instance
_document_loader_factory = None


def get_document_loader() -> DocumentLoaderFactory:
    """Get singleton document loader factory"""
    global _document_loader_factory
    if _document_loader_factory is None:
        _document_loader_factory = DocumentLoaderFactory()
    return _document_loader_factory
