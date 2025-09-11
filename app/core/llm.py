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
import re

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
            try:
                return ChatGoogleGenerativeAI(
                    model=settings.llm_model,
                    google_api_key=settings.gemini_api_key,
                    temperature=settings.llm_temperature,
                    max_output_tokens=settings.llm_max_tokens,
                    max_retries=3
                )
            except Exception as e:
                logger.warning(f"Failed to initialize {settings.llm_model}, falling back to gemini-1.5-flash", error=str(e))
                return ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=settings.gemini_api_key,
                    temperature=settings.llm_temperature,
                    max_output_tokens=settings.llm_max_tokens,
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
                max_retries=3
            )
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate text response using LangChain"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            # Filter out kwargs that might not be supported by the specific model
            safe_kwargs = {}
            for key, value in kwargs.items():
                if key not in ['temperature', 'max_tokens', 'timeout']:  # Skip problematic params
                    safe_kwargs[key] = value
            
            response = await self.model.ainvoke(messages, **safe_kwargs)
            return response.content
        except Exception as e:
            logger.error("LLM generation failed", error=str(e), model=settings.llm_model)
            raise
    
    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences and extract JSON body if present."""
        if not text:
            return text
        # Match ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()
        return text.strip()

    def _coerce_to_json(self, raw: str) -> Dict[str, Any]:
        """Best-effort conversion of model output to JSON dict."""
        raw = (raw or "").strip()
        # First, strip code fences
        candidate = self._strip_code_fences(raw)
        # Try direct json
        try:
            return json.loads(candidate)
        except Exception:
            pass
        # Extract first {...} block
        try:
            start = candidate.find('{')
            end = candidate.rfind('}')
            if start != -1 and end != -1 and end > start:
                inner = candidate[start:end+1]
                return json.loads(inner)
        except Exception:
            pass
        # As a last resort, replace single quotes and trailing commas (very naive)
        naive = candidate.replace("'", '"')
        naive = re.sub(r",\s*([}\]])", r"\1", naive)
        return json.loads(naive)

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
        enhanced_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema (no comments, no prose, no code fences):\n"
            f"{json.dumps(schema, indent=2)}\n"
            f"Output ONLY the JSON object."
        )
        messages.append(HumanMessage(content=enhanced_prompt))
        
        try:
            # Avoid provider-specific structured output features that can fail (e.g., Gemini tools).
            # Ask for JSON in the prompt and parse the model output.
            # Filter out kwargs that might not be supported by the specific model
            safe_kwargs = {}
            for key, value in kwargs.items():
                if key not in ['temperature', 'max_tokens', 'timeout']:  # Skip problematic params
                    safe_kwargs[key] = value
            
            response = await self.model.ainvoke(messages, **safe_kwargs)
            try:
                # First try the built-in parser for strictness
                return self.json_parser.parse(response.content)
            except Exception:
                # Fallback to tolerant parsing
                return self._coerce_to_json(response.content)
        except Exception as e:
            logger.error("JSON generation failed", error=str(e), model=settings.llm_model)
            # Attempt a single repair round-trip with a strict instruction
            try:
                repair_messages = []
                if system_prompt:
                    repair_messages.append(SystemMessage(content=system_prompt))
                repair_messages.append(HumanMessage(content=(
                    "Return ONLY a valid JSON that matches the schema above. "
                    "Do not include any markdown or commentary. "
                    "Fix and output the JSON for this content:\n\n" + (enhanced_prompt)
                )))
                # Use same safe kwargs filter for repair
                safe_kwargs = {}
                for key, value in kwargs.items():
                    if key not in ['temperature', 'max_tokens', 'timeout']:
                        safe_kwargs[key] = value
                
                repaired = await self.model.ainvoke(repair_messages, **safe_kwargs)
                return self._coerce_to_json(repaired.content)
            except Exception as e2:
                logger.error("JSON repair failed", error=str(e2), model=settings.llm_model)
                raise
    
    async def stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> AsyncIterator[str]:
        """Stream text response"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            # Filter out kwargs that might not be supported by the specific model
            safe_kwargs = {}
            for key, value in kwargs.items():
                if key not in ['temperature', 'max_tokens', 'timeout']:  # Skip problematic params
                    safe_kwargs[key] = value
            
            async for chunk in self.model.astream(messages, **safe_kwargs):
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

You decide how many modules and lessons are appropriate based on the topic and resources.
CONSTRAINTS:
- Minimum 3 modules total.
- Minimum 2 lessons in every module.
- Do NOT over-generate filler; if there is nothing meaningful to add, stop at a reasonable number.
- Prefer 3–8 modules and 2–6 lessons per module when justified by content.
- Avoid duplicate or placeholder titles; each title must be specific and content-based.
- Each lesson must include a concise, non-empty description.

Generate a well-structured course that covers the important topics with logical progression from fundamentals to advanced concepts.
Include learning objectives implicitly within lesson descriptions when relevant. Keep the JSON concise and compliant with the schema (use 'order' fields)."""

        system_prompt = "You are an expert curriculum designer specializing in creating effective learning paths."
        
        try:
            route = await self.llm.generate_json(prompt, schema, system_prompt)

            # Normalize + fill required fields, ensure required titles exist
            import uuid
            modules = route.get("modules", []) or []
            for mi, module in enumerate(modules, start=1):
                if not module.get("module_id"):
                    module["module_id"] = str(uuid.uuid4())
                # Ensure 'order'
                if "order" not in module and "position" in module:
                    module["order"] = module.get("position")
                module.setdefault("order", mi)
                # Ensure module title
                m_title = module.get("title") or module.get("name") or module.get("heading")
                if not m_title or not str(m_title).strip():
                    module["title"] = f"Module {mi}"
                else:
                    module["title"] = str(m_title).strip()
                # Ensure module description (non-blank for downstream validation)
                m_desc = module.get("description")
                if not m_desc or not str(m_desc).strip():
                    module["description"] = f"Overview of {module['title']}."

                lessons = module.get("lessons", []) or []
                normalized_lessons = []
                for li, lesson in enumerate(lessons, start=1):
                    if not isinstance(lesson, dict):
                        # Skip invalid lesson entries
                        continue
                    if not lesson.get("lesson_id"):
                        lesson["lesson_id"] = str(uuid.uuid4())
                    # Ensure 'order'
                    if "order" not in lesson and "position" in lesson:
                        lesson["order"] = lesson.get("position")
                    lesson.setdefault("order", li)
                    # Ensure lesson title
                    l_title = lesson.get("title") or lesson.get("name") or lesson.get("heading")
                    if not l_title or not str(l_title).strip():
                        # Try derive from description first
                        desc = str(lesson.get("description", "")).strip()
                        if desc:
                            words = desc.split()
                            candidate = " ".join(words[:6]).strip()
                            lesson["title"] = candidate if candidate else f"Lesson {li}"
                        else:
                            lesson["title"] = f"Lesson {li}"
                    else:
                        lesson["title"] = str(l_title).strip()
                    # Ensure non-blank description
                    l_desc = lesson.get("description")
                    if not l_desc or not str(l_desc).strip():
                        lesson["description"] = f"This lesson covers {lesson['title']}."
                    else:
                        lesson["description"] = str(l_desc).strip()
                    # Ensure min mastery
                    lesson["min_mastery"] = 0.65
                    normalized_lessons.append(lesson)

                module["lessons"] = normalized_lessons

            logger.info("Course route generated", 
                       course_id=course_id,
                       modules=len(route.get("modules", [])))
            
            return route
        except Exception as e:
            logger.error("Route generation failed", error=str(e), course_id=course_id)
            raise

    async def generate_lesson_content(
        self,
        course_title: str,
        course_description: str,
        lesson_title: str,
        language: str = "en",
        target_length_words: int = 800
    ) -> str:
        """Generate standalone lesson content without using RAG"""
        system_prompt = (
            "You are an expert educator. Create high-quality lesson content "
            "for the specified course and lesson in the requested language. "
            "Content must be self-contained (do not assume external materials)."
        )

        user_prompt = (
            f"Course: {course_title}\n"
            f"Description: {course_description}\n"
            f"Lesson: {lesson_title}\n"
            f"Language: {language}\n\n"
            "Write comprehensive lesson content with:\n"
            "- Clear introduction and learning objectives\n"
            "- Main sections with explanations and examples\n"
            "- Code snippets or tables when relevant\n"
            "- A short summary and 3–5 practice questions\n"
            f"Aim for about {target_length_words}–{int(target_length_words*1.5)} words."
        )

        try:
            return (await self.llm.generate(user_prompt, system_prompt=system_prompt)).strip()
        except Exception as e:
            logger.error("Lesson content generation failed", error=str(e), lesson=lesson_title)
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
