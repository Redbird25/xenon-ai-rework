"""
Resource fetching utilities with support for various document types
"""
import pathlib
from typing import Dict, Any

from app.core.document_loaders import get_document_loader
from app.core.logging import get_logger

logger = get_logger(__name__)


async def fetch_resource(resource: str) -> str:
    """
    Legacy function for backward compatibility.
    Fetches content from resource and returns plain text.
    """
    document = await fetch_resource_with_metadata(resource)
    return document.get('content', '')


async def fetch_resource_with_metadata(resource: str) -> Dict[str, Any]:
    """
    Fetches content from resource with metadata.
    Supports various document types: PDF, HTML, Markdown, etc.
    
    Returns:
        Dict with keys: content, source, document_type, metadata, content_hash
    """
    # Handle file:// URLs
    if resource.startswith("file://"):
        resource = str(pathlib.Path(resource[7:]).absolute())
    
    # Use document loader factory
    loader = get_document_loader()
    document = await loader.load_document(resource)
    
    logger.info("Resource fetched",
               source=resource,
               document_type=document.get('document_type'),
               content_length=len(document.get('content', '')))
    
    return document


async def fetch_resources(resources: list[str]) -> list[Dict[str, Any]]:
    """
    Fetch multiple resources concurrently.
    
    Returns:
        List of document dictionaries with content and metadata
    """
    loader = get_document_loader()
    documents = await loader.load_documents(resources)
    
    logger.info("Multiple resources fetched",
               total=len(resources),
               successful=len(documents))
    
    return documents
