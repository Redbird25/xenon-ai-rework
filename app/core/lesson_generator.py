"""
Lesson generation service with chunk search and LLM integration
"""
import uuid
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog

from app.config import settings
from app.core.vector_search import get_search_engine
from app.core.llm import get_llm_provider
from app.db import async_session
from app.models import MaterializedLesson, IngestJob, LessonChunk
from app.schemas import UserPreferences, MaterializedLesson as MaterializedLessonSchema, LessonSection
from sqlalchemy import select, update

logger = structlog.get_logger(__name__)


@dataclass
class LessonGenerationContext:
    """Context for lesson generation"""
    lesson_name: str
    description: str
    user_preferences: UserPreferences
    source_chunks: List[Dict[str, Any]]
    course_id: str
    content_strategy: str  # 'resource_rich', 'resource_mixed', 'resource_none'


class LessonGenerator:
    """Service for generating materialized lessons"""
    
    def __init__(self):
        self.search_engine = get_search_engine()
        self.llm_provider = get_llm_provider()
    
    async def generate_lesson(
        self, 
        course_id: str, 
        lesson_name: str, 
        description: str, 
        user_preferences: UserPreferences,
        job_id: str
    ) -> MaterializedLessonSchema:
        """Generate a complete lesson from course content"""
        
        start_time = time.time()
        logger.info(
            "Starting lesson generation",
            job_id=job_id,
            course_id=course_id,
            lesson_name=lesson_name
        )
        
        try:
            # Update job status to processing
            await self._update_job_status(job_id, "processing")
            
            # Step 1: Search for relevant chunks and determine strategy
            relevant_chunks, content_strategy = await self._find_relevant_chunks(
                course_id, lesson_name, description
            )
            
            logger.info(
                "Resource analysis completed",
                job_id=job_id,
                chunks_count=len(relevant_chunks),
                content_strategy=content_strategy
            )
            
            # Step 2: Create generation context
            context = LessonGenerationContext(
                lesson_name=lesson_name,
                description=description,
                user_preferences=user_preferences,
                source_chunks=relevant_chunks,
                course_id=course_id,
                content_strategy=content_strategy
            )
            
            # Step 3: Generate lesson content with LLM
            lesson_content = await self._generate_lesson_content(context)
            
            # Step 4: Save materialized lesson
            lesson_id = await self._save_materialized_lesson(
                context, lesson_content, job_id
            )
            
            # Step 5: Update job status to completed
            generation_time = time.time() - start_time
            await self._update_job_status(
                job_id, 
                "completed", 
                {
                    "lesson_id": str(lesson_id),
                    "chunks_used": len(relevant_chunks),
                    "generation_time_seconds": generation_time
                }
            )
            
            logger.info(
                "Lesson generation completed",
                job_id=job_id,
                lesson_id=str(lesson_id),
                generation_time=generation_time
            )
            
            # Return the saved lesson from database instead of pydantic model
            async with async_session() as session:
                saved_lesson = await session.get(MaterializedLesson, lesson_id)
                # Add the content strategy as attribute for logging
                if saved_lesson:
                    saved_lesson._content_strategy = context.content_strategy
                    saved_lesson.generated_from_chunks = context.source_chunks
                return saved_lesson or lesson_content
            
        except Exception as e:
            logger.error(
                "Lesson generation failed",
                job_id=job_id,
                error=str(e),
                lesson_name=lesson_name
            )
            await self._update_job_status(job_id, "failed", {"error": str(e)})
            raise
    
    async def _find_relevant_chunks(
        self, 
        course_id: str, 
        lesson_name: str, 
        description: str
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Find relevant chunks for lesson generation.
        Returns: (chunks, content_strategy)
        
        Content strategies:
        - 'resource_rich': много ресурсов, генерим только по ним
        - 'resource_mixed': мало ресурсов, миксуем с LLM генерацией
        - 'resource_none': нет ресурсов, полная LLM генерация + инжест
        """
        
        # Create search query combining lesson name and description
        search_query = f"{lesson_name}. {description}"
        
        # Perform hybrid search with expanded results for analysis
        search_results = await self.search_engine.search(
            query=search_query,
            course_id=course_id,
            top_k=15,  # More results for better analysis
            similarity_threshold=0.5,  # Lower threshold for broader search
            use_hybrid=True
        )
        
        # Convert search results to format needed for generation
        chunks = []
        high_relevance_chunks = 0
        
        for result in search_results:
            chunks.append({
                "id": result.chunk_id,
                "content": result.content,
                "source_ref": result.source_ref,
                "metadata": result.metadata,
                "similarity_score": result.similarity_score
            })
            
            # Count high relevance chunks (similarity > 0.7)
            if result.similarity_score > 0.7:
                high_relevance_chunks += 1
        
        # Determine content strategy based on available resources
        total_chunks = len(chunks)
        if total_chunks == 0:
            content_strategy = "resource_none"
        elif total_chunks < 3 or high_relevance_chunks < 2:
            content_strategy = "resource_mixed" 
        else:
            content_strategy = "resource_rich"
            # Limit chunks for resource_rich to prevent overwhelming LLM
            chunks = chunks[:8]
        
        logger.info(
            "Resource analysis completed",
            total_chunks=total_chunks,
            high_relevance_chunks=high_relevance_chunks,
            content_strategy=content_strategy
        )
        
        return chunks, content_strategy
    
    async def _generate_lesson_content(
        self, 
        context: LessonGenerationContext
    ) -> MaterializedLessonSchema:
        """Generate lesson content using LLM"""
        
        # Create comprehensive prompt for lesson generation
        prompt = self._create_lesson_prompt(context)
        
        # Define JSON schema for structured output
        lesson_schema = {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "examples": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["title", "content", "examples"]
                    }
                }
            },
            "required": ["sections"]
        }
        
        # Generate structured content with higher token limit for rich lessons
        response = await self.llm_provider.generate_json(
            prompt=prompt,
            schema=lesson_schema,
            temperature=0.4,  # More creative for diverse content
            max_tokens=32000  # Increased for comprehensive lessons
        )
        
        # Parse and validate response
        sections = []
        for section_data in response.get("sections", []):
            section = LessonSection(
                title=section_data["title"],
                content=section_data["content"],
                examples=section_data["examples"]
            )
            sections.append(section)
        
        # Create materialized lesson
        lesson = MaterializedLessonSchema(
            lesson_name=context.lesson_name,
            description=context.description,
            sections=sections,
            generated_from_chunks=[chunk["id"] for chunk in context.source_chunks]
        )
        
        # Add content strategy as an attribute for later use
        lesson._content_strategy = context.content_strategy
        
        return lesson
    
    def _create_lesson_prompt(self, context: LessonGenerationContext) -> str:
        """Create detailed prompt for lesson generation based on content strategy"""
        
        # Format user preferences
        interests_str = ", ".join(context.user_preferences.interests)
        hobbies_str = ", ".join(context.user_preferences.hobbies)
        learning_style = context.user_preferences.learning_style
        
        # Create strategy-specific instructions
        if context.content_strategy == "resource_rich":
            # Много ресурсов - создаем богатый урок на их основе
            source_content = []
            for i, chunk in enumerate(context.source_chunks):
                source_content.append(f"Source {i+1}: {chunk['content']}")
            source_text = "\n\n".join(source_content)
            
            strategy_instructions = """
CONTENT STRATEGY - RESOURCE RICH:
- You have abundant high-quality source materials for this lesson
- Create a comprehensive, well-structured lesson based on these materials
- SYNTHESIZE and EXPAND on the source materials to create engaging educational content
- Add detailed explanations, step-by-step breakdowns, and practical applications
- Create original examples that illustrate the concepts from source materials
- Make the lesson thorough and educational - aim for 5-7 substantial sections
- Personalize examples to student interests while maintaining academic integrity
- Focus on deep understanding and practical application of concepts
"""
            
        elif context.content_strategy == "resource_mixed":
            # Мало ресурсов - миксуем
            source_content = []
            for chunk in context.source_chunks:
                source_content.append(f"Content: {chunk['content'][:500]}...")
            source_text = "\n\n".join(source_content)
            
            strategy_instructions = """
CONTENT STRATEGY - RESOURCE MIXED:
- You have limited source materials for this lesson
- Use provided sources as the foundation, but you may supplement with standard educational content
- When adding content, clearly distinguish between source-based and supplemental information
- Prioritize source materials, but fill gaps with well-established educational principles
- Examples can combine source material with generated examples that align with student interests
"""
            
        else:  # resource_none
            # Нет ресурсов - полная генерация
            source_text = "No specific source materials available for this lesson."
            
            strategy_instructions = f"""
CONTENT STRATEGY - RESOURCE NONE:
- No specific source materials are available for this lesson
- Generate comprehensive educational content based on standard curriculum for this topic
- Ensure all content is academically sound and follows educational best practices
- MANDATORY: Create original examples that connect to student's interests: {interests_str} AND hobbies: {hobbies_str}
- EVERY example must combine technical interests with personal hobbies
- Use real-world scenarios from software development industry that relate to their hobbies
- Connect concepts to practical programming applications in contexts they care about
- This generated content will be added to the course knowledge base

PERSONALIZATION FOCUS FOR THIS STRATEGY:
Since you're creating content from scratch, you have FULL FREEDOM to make every example
relevant to the student's interests ({interests_str}) combined with their hobbies ({hobbies_str}).
Create unique combinations like "AI for gaming", "Backend for music apps", "Frontend for sports dashboards".
"""
        
        # Add timestamp for variety in generation
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        prompt = f"""
You are an expert educational content creator specializing in personalized programming education.

🚨 CRITICAL REQUIREMENT: This lesson MUST be heavily personalized for this specific student's interests.

LESSON DETAILS:
- Name: {context.lesson_name}
- Description: {context.description}
- Generation Time: {current_time}

STUDENT PROFILE:
- Interests: {interests_str}
- Hobbies: {hobbies_str}
- Learning Style: {learning_style}

{strategy_instructions}

SOURCE MATERIALS:
{source_text}

CRITICAL PERSONALIZATION REQUIREMENTS:
🎯 STUDENT INTERESTS: {interests_str}
🎯 STUDENT HOBBIES: {hobbies_str}
🎯 LEARNING STYLE: {learning_style}

MANDATORY PERSONALIZATION RULES:
1. EVERY example must relate to student's interests ({interests_str}) AND/OR hobbies ({hobbies_str})
2. Use specific technical contexts from their interests in examples
3. Connect lesson concepts to their career/hobby goals
4. Reference real tools/frameworks they would use (React, Node.js, Python, ML libraries, etc.)
5. Combine interests + hobbies creatively (e.g., AI for gaming, Backend for music apps, etc.)

INSTRUCTIONS:
1. Create a comprehensive, well-structured lesson with 5-7 substantial sections
2. Each section MUST have:
   - Clear, descriptive title
   - Rich, detailed content explanation (aim for 300-500 words per section)  
   - Step-by-step breakdowns where appropriate
   - 3-4 practical examples that DIRECTLY connect to their interests ({interests_str}) and hobbies ({hobbies_str})
   - Real-world applications from software development industry
3. Adapt content presentation based on learning style:
   - TEXT: Focus on detailed explanations, code examples, and structured information
   - VIDEO: Include descriptions of visual concepts, diagrams, and step-by-step processes
   - MIXED: Balance both approaches with varied presentation methods
4. Follow the content strategy guidelines above strictly
5. MANDATORY: Examples must combine interests + hobbies contexts like:
   - AI + Gaming: Neural networks for game AI, player behavior prediction
   - Backend + Music: Music streaming APIs, playlist algorithms, audio processing
   - Frontend + Sports: Sports statistics dashboards, real-time score updates
   - AI + Photography: Image recognition, photo enhancement algorithms
   - Backend + Travel: Travel booking systems, location APIs, route optimization
   - Frontend + Social: Social media interfaces, chat applications, user interactions
6. Include practical exercises using modern dev tools and frameworks
7. Add variety - each section should feel unique and valuable
8. Ensure progression from basic concepts to more advanced applications

Generate a rich, personalized lesson in the specified JSON format with sections array.
"""
        
        return prompt.strip()
    
    async def _save_materialized_lesson(
        self, 
        context: LessonGenerationContext, 
        lesson_content: MaterializedLessonSchema,
        job_id: str
    ) -> uuid.UUID:
        """Save generated lesson to database"""
        
        lesson_id = uuid.uuid4()
        
        async with async_session() as session:
            # Add content strategy to metadata
            content_data = lesson_content.model_dump()
            content_data["content_strategy"] = context.content_strategy
            
            materialized_lesson = MaterializedLesson(
                id=lesson_id,
                course_id=context.course_id,
                lesson_name=context.lesson_name,
                description=context.description,
                content=content_data,
                user_preferences=context.user_preferences.model_dump(),
                source_chunks=[chunk["id"] for chunk in context.source_chunks],
                generation_model=settings.llm_model,
                processing_status="completed",
                job_id=job_id
            )
            
            session.add(materialized_lesson)
            await session.commit()
            
        return lesson_id
    
    async def _update_job_status(
        self, 
        job_id: str, 
        status: str, 
        result_data: Optional[Dict[str, Any]] = None
    ):
        """Update job status in database"""
        
        async with async_session() as session:
            update_data = {"status": status}
            
            if status == "processing":
                update_data["started_at"] = datetime.utcnow()
            elif status in ["completed", "failed"]:
                update_data["completed_at"] = datetime.utcnow()
                if result_data:
                    update_data["result_data"] = result_data
            
            await session.execute(
                update(IngestJob)
                .where(IngestJob.id == job_id)
                .values(**update_data)
            )
            await session.commit()


# Singleton instance
_lesson_generator = None


def get_lesson_generator() -> LessonGenerator:
    """Get singleton lesson generator instance"""
    global _lesson_generator
    if _lesson_generator is None:
        _lesson_generator = LessonGenerator()
    return _lesson_generator