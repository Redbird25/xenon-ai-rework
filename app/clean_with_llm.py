"""
Legacy document cleaning module for backward compatibility
"""
from app.core.llm import get_document_cleaner
import asyncio


def clean_with_llm(raw_text: str, lang: str = "en") -> str:
    """
    Legacy synchronous function for cleaning documents.
    Uses the new DocumentCleaner under the hood.
    """
    cleaner = get_document_cleaner()
    
    # Run async function in sync context
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # If there's already a running loop, we need to run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, cleaner.clean_document(raw_text, doc_type="web"))
                return future.result()
        except RuntimeError:
            # No running loop, we can use asyncio.run
            return asyncio.run(cleaner.clean_document(raw_text, doc_type="web"))
    except Exception as e:
        print(f"Error cleaning document: {e}")
        return ""
