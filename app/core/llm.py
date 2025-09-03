"""
Modern LLM module using LangChain with support for multiple providers
"""
from typing import Dict, Any, Optional, List, AsyncIterator
from abc import ABC, abstractmethod
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.callbacks import AsyncCallbackHandler
from langchain.schema import Document
from pydantic import BaseModel, Field
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response"""
        pass
    
    @abstractmethod
    async def generate_json(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured JSON response"""
        pass
    
    @abstractmethod
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream text response"""
        pass


class LangChainLLMProvider(LLMProvider):
    """LangChain-based LLM provider with multi-model support"""
    
    def __init__(self):
        self.model = self._initialize_model()
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()
        
    def _initialize_model(self):
        """Initialize the appropriate LLM based on configuration"""
        if "gemini" in settings.llm_model.lower():
            return ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=settings.gemini_api_key,
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout,
                max_retries=3
            )
        elif "gpt" in settings.llm_model.lower() and settings.openai_api_key:
            return ChatOpenAI(
                model=settings.llm_model,
                openai_api_key=settings.openai_api_key,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout,
                max_retries=3
            )
        elif "claude" in settings.llm_model.lower() and settings.anthropic_api_key:
            return ChatAnthropic(
                model=settings.llm_model,
                anthropic_api_key=settings.anthropic_api_key,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout,
                max_retries=3
            )
        else:
            # Default to Gemini
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=settings.gemini_api_key,
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout
            )
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate text response using LangChain"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = await self.model.ainvoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error("LLM generation failed", error=str(e), model=settings.llm_model)
            raise
    
    async def generate_json(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured JSON response"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Add schema instruction to prompt
        enhanced_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        messages.append(HumanMessage(content=enhanced_prompt))
        
        try:
            # Avoid provider-specific structured output features that can fail (e.g., Gemini tools).
            # Ask for JSON in the prompt and parse the model output.
            response = await self.model.ainvoke(messages, **kwargs)
            return self.json_parser.parse(response.content)
        except Exception as e:
            logger.error("JSON generation failed", error=str(e), model=settings.llm_model)
            raise
    
    async def stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> AsyncIterator[str]:
        """Stream text response"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            async for chunk in self.model.astream(messages, **kwargs):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error("LLM streaming failed", error=str(e), model=settings.llm_model)
            raise


class DocumentCleaner:
    """Clean and preprocess documents using LLM"""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or LangChainLLMProvider()
        self.cleaning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document cleaning assistant. Your task is to:
1. Remove all navigation elements, headers, footers, ads, and UI components
2. Extract only the main content (articles, tutorials, documentation)
3. Preserve code blocks, tables, and important formatting
4. Return clean, well-formatted text suitable for learning

Output only the cleaned content without any explanations."""),
            ("human", "Clean this document:\n\n{document}")
        ])
    
    async def clean_document(self, raw_text: str, doc_type: str = "web") -> str:
        """Clean raw document text"""
        try:
            # Truncate if too long
            if len(raw_text) > 12000:
                raw_text = raw_text[:12000] + "..."
            
            prompt = self.cleaning_prompt.format(document=raw_text)
            cleaned = await self.llm.generate(str(prompt))
            
            logger.info("Document cleaned", 
                       original_length=len(raw_text), 
                       cleaned_length=len(cleaned),
                       doc_type=doc_type)
            
            return cleaned.strip()
        except Exception as e:
            logger.error("Document cleaning failed", error=str(e))
            # Return original if cleaning fails
            return raw_text


class CourseRouteGenerator:
    """Generate course structure using LLM"""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or LangChainLLMProvider()
    
    async def generate_route(
        self,
        course_id: str,
        title: str,
        description: str,
        resources: List[str],
        language: str = "en"
    ) -> Dict[str, Any]:
        """Generate course route structure"""
        
        # Define the schema for course structure to match callback contract (using 'order')
        schema = {
            "type": "object",
            "properties": {
                "modules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "module_id": {"type": "string"},
                            "title": {"type": "string"},
                            "order": {"type": "integer"},
                            "description": {"type": "string"},
                            "lessons": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "lesson_id": {"type": "string"},
                                        "title": {"type": "string"},
                                        "description": {"type": "string"},
                                        "order": {"type": "integer"},
                                        "min_mastery": {"type": "number"}
                                    },
                                    "required": ["lesson_id", "title", "description", "order", "min_mastery"]
                                }
                            }
                        },
                        "required": ["module_id", "title", "order", "lessons"]
                    }
                }
            },
            "required": ["modules"]
        }
        
        prompt = f"""Create a comprehensive course structure for:
Course ID: {course_id}
Title: {title}
Description: {description}
Resources: {', '.join(resources)}
Language: {language}

Generate a well-structured course with modules and lessons that covers all important topics.
Include learning objectives, prerequisites, and difficulty levels.
Ensure logical progression from basic to advanced concepts."""

        system_prompt = "You are an expert curriculum designer specializing in creating effective learning paths."
        
        try:
            route = await self.llm.generate_json(prompt, schema, system_prompt)
            
            # Normalize + fill required fields
            import uuid
            for module in route.get("modules", []):
                if not module.get("module_id"):
                    module["module_id"] = str(uuid.uuid4())
                # ensure 'order'
                if "order" not in module and "position" in module:
                    module["order"] = module.get("position")
                module.setdefault("order", 1)
                for lesson in module.get("lessons", []):
                    if not lesson.get("lesson_id"):
                        lesson["lesson_id"] = str(uuid.uuid4())
                    if "order" not in lesson and "position" in lesson:
                        lesson["order"] = lesson.get("position")
                    lesson.setdefault("order", 1)
                    lesson.setdefault("description", "")
                    # Always enforce constant min_mastery
                    lesson["min_mastery"] = 0.65
            
            logger.info("Course route generated", 
                       course_id=course_id,
                       modules=len(route.get("modules", [])))
            
            return route
        except Exception as e:
            logger.error("Route generation failed", error=str(e), course_id=course_id)
            raise


class RAGQueryProcessor:
    """Process queries for RAG system"""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or LangChainLLMProvider()
        self.query_enhancement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query enhancement assistant. Your task is to:
1. Expand user queries with relevant synonyms and related terms
2. Identify key concepts and entities
3. Generate alternative phrasings
4. Extract search keywords

Return a JSON with: {
    "enhanced_query": "expanded query with more context",
    "keywords": ["key", "search", "terms"],
    "concepts": ["main", "concepts"],
    "filters": {"language": "en", "difficulty": "beginner"}
}"""),
            ("human", "Enhance this query: {query}")
        ])
    
    async def enhance_query(self, query: str) -> Dict[str, Any]:
        """Enhance user query for better search results"""
        try:
            # Use structured JSON generation for robustness
            schema = {
                "type": "object",
                "properties": {
                    "enhanced_query": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "concepts": {"type": "array", "items": {"type": "string"}},
                    "filters": {"type": "object"}
                },
                "required": ["enhanced_query", "keywords", "concepts", "filters"]
            }

            system_prompt = (
                "You are a query enhancement assistant. Your task is to:\n"
                "1. Expand user queries with relevant synonyms and related terms\n"
                "2. Identify key concepts and entities\n"
                "3. Generate alternative phrasings\n"
                "4. Extract search keywords\n"
                "Return only a compact JSON object."
            )

            user_prompt = f"Enhance this query: {query}"

            result = await self.llm.generate_json(user_prompt, schema, system_prompt)

            logger.info("Query enhanced", 
                       original=query,
                       keywords=len(result.get("keywords", [])))

            return result
        except Exception as e:
            logger.error("Query enhancement failed", error=str(e))
            # Return original query if enhancement fails
            return {
                "enhanced_query": query,
                "keywords": query.split(),
                "concepts": [],
                "filters": {}
            }


# Singleton instances
_llm_provider = None
_document_cleaner = None
_course_generator = None
_query_processor = None


def get_llm_provider() -> LLMProvider:
    """Get singleton LLM provider instance"""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = LangChainLLMProvider()
    return _llm_provider


def get_document_cleaner() -> DocumentCleaner:
    """Get singleton document cleaner instance"""
    global _document_cleaner
    if _document_cleaner is None:
        _document_cleaner = DocumentCleaner()
    return _document_cleaner


def get_course_generator() -> CourseRouteGenerator:
    """Get singleton course generator instance"""
    global _course_generator
    if _course_generator is None:
        _course_generator = CourseRouteGenerator()
    return _course_generator


def get_query_processor() -> RAGQueryProcessor:
    """Get singleton query processor instance"""
    global _query_processor
    if _query_processor is None:
        _query_processor = RAGQueryProcessor()
    return _query_processor
