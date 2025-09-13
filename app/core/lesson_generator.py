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
from app.schemas import UserPreferences, MaterializedLesson as MaterializedLessonSchema, LessonSection, CodeExample, PracticalExercise
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
                    saved_lesson.generated_from_chunks = [chunk["id"] if isinstance(chunk, dict) else chunk for chunk in context.source_chunks]
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
    
    def _is_technical_topic(self, lesson_name: str, description: str) -> bool:
        """Determine if lesson is about programming/technical topics"""
        technical_keywords = [
            # Programming languages
            'javascript', 'python', 'java', 'react', 'node.js', 'typescript', 'html', 'css',
            'php', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'c++', 'c#', 'sql',
            # Frameworks & Libraries
            'react', 'vue', 'angular', 'django', 'flask', 'express', 'spring', 'laravel',
            'bootstrap', 'tailwind', 'jquery', 'redux', 'nextjs', 'nuxt',
            # Technologies & Concepts
            'api', 'rest', 'graphql', 'database', 'mysql', 'postgresql', 'mongodb',
            'docker', 'kubernetes', 'aws', 'cloud', 'microservices', 'devops',
            'algorithm', 'data structure', 'oop', 'functional programming',
            'machine learning', 'ai', 'neural network', 'deep learning',
            'frontend', 'backend', 'fullstack', 'web development', 'mobile development',
            'git', 'github', 'version control', 'testing', 'unit test', 'integration test',
            'hooks', 'component', 'state management', 'props', 'jsx', 'dom',
            'async', 'await', 'promise', 'callback', 'event', 'debugging'
        ]

        text_to_check = f"{lesson_name} {description}".lower()
        return any(keyword in text_to_check for keyword in technical_keywords)

    async def _generate_lesson_content(
        self,
        context: LessonGenerationContext
    ) -> MaterializedLessonSchema:
        """Generate lesson content using LLM with enhanced support for technical topics"""

        # Determine if this is a technical lesson
        is_technical = self._is_technical_topic(context.lesson_name, context.description)

        # Create comprehensive prompt for lesson generation
        prompt = self._create_lesson_prompt(context, is_technical)

        # Define enhanced JSON schema for technical lessons
        if is_technical:
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
                                "code_examples": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "language": {"type": "string"},
                                            "code": {"type": "string"},
                                            "explanation": {"type": "string"},
                                            "context": {"type": "string"}
                                        },
                                        "required": ["language", "code", "explanation", "context"]
                                    }
                                },
                                "practical_exercises": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "task": {"type": "string"},
                                            "solution_hint": {"type": "string"},
                                            "difficulty": {"type": "string"}
                                        },
                                        "required": ["task", "solution_hint", "difficulty"]
                                    }
                                },
                                "examples": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["title", "content", "code_examples", "practical_exercises", "examples"]
                        }
                    }
                },
                "required": ["sections"]
            }
        else:
            # Original schema for non-technical lessons
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
        
        # Parse and validate response with enhanced technical content support
        sections = []
        for i, section_data in enumerate(response.get("sections", []), 1):
            section_title = section_data.get("title", f"Section {i}")
            section_content = section_data.get("content", "")
            section_examples = section_data.get("examples", [])

            if is_technical:
                # For technical lessons, validate and include code examples
                code_examples = section_data.get("code_examples", [])
                practical_exercises = section_data.get("practical_exercises", [])

                # Validate technical content - ensure we have code examples
                if not code_examples:
                    logger.warning(f"Technical section '{section_title}' missing code examples, generating placeholder")
                    # Generate a basic code example as fallback
                    code_examples = [{
                        "language": "javascript",
                        "code": f"// Example code for {section_title}\nconsole.log('This section needs proper code examples');",
                        "explanation": "This is a placeholder code example. The lesson generator should provide actual code examples for technical topics.",
                        "context": f"Technical example for {section_title}"
                    }]

                if not practical_exercises:
                    logger.warning(f"Technical section '{section_title}' missing exercises, generating placeholder")
                    practical_exercises = [{
                        "task": f"Practice implementing concepts from {section_title}",
                        "solution_hint": "Follow the examples provided in this section",
                        "difficulty": "intermediate"
                    }]

                # Convert to Pydantic models
                pydantic_code_examples = [
                    CodeExample(
                        language=ce.get("language", "javascript"),
                        code=ce.get("code", ""),
                        explanation=ce.get("explanation", ""),
                        context=ce.get("context", "")
                    ) for ce in code_examples
                ]

                pydantic_exercises = [
                    PracticalExercise(
                        task=ex.get("task", ""),
                        solution_hint=ex.get("solution_hint", ""),
                        difficulty=ex.get("difficulty", "intermediate")
                    ) for ex in practical_exercises
                ]

                section = LessonSection(
                    title=section_title,
                    content=section_content,
                    examples=section_examples,
                    code_examples=pydantic_code_examples,
                    practical_exercises=pydantic_exercises
                )
                sections.append(section)

                # Log technical content validation
                logger.info(
                    f"Technical section validated: {section_title}",
                    code_examples_count=len(code_examples),
                    exercises_count=len(practical_exercises)
                )
            else:
                # For non-technical lessons, use original format
                section = LessonSection(
                    title=section_title,
                    content=section_content,
                    examples=section_examples
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
    
    def _create_lesson_prompt(self, context: LessonGenerationContext, is_technical: bool = False) -> str:
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
🎯 LESSON TYPE: {"TECHNICAL/PROGRAMMING" if is_technical else "GENERAL EDUCATION"}

MANDATORY PERSONALIZATION RULES:
1. EVERY example must relate to student's interests ({interests_str}) AND/OR hobbies ({hobbies_str})
2. Use specific technical contexts from their interests in examples
3. Connect lesson concepts to their career/hobby goals
4. Reference real tools/frameworks they would use (React, Node.js, Python, ML libraries, etc.)
5. Combine interests + hobbies creatively (e.g., AI for gaming, Backend for music apps, etc.)

{"🚨 CRITICAL FOR TECHNICAL LESSONS: This is a programming/technical topic - you MUST include practical code examples!" if is_technical else ""}

{f'''🔥 TECHNICAL LESSON JSON STRUCTURE EXAMPLE:
{{
  "sections": [
    {{
      "title": "React useState Hook Basics",
      "content": "The useState hook allows you to add state to functional components...",
      "code_examples": [
        {{
          "language": "javascript",
          "code": "import React, {{ useState }} from 'react';\\n\\nfunction GameScore() {{\\n  const [score, setScore] = useState(0);\\n\\n  return (\\n    <div>\\n      <p>Score: {{score}}</p>\\n      <button onClick={{() => setScore(score + 1)}}>\\n        Add Point\\n      </button>\\n    </div>\\n  );\\n}}",
          "explanation": "This example shows a gaming score tracker using useState. The score starts at 0 and increases when the button is clicked.",
          "context": "Gaming score tracker for {interests_str[0] if interests_str else 'your hobby'}"
        }},
        {{
          "language": "javascript",
          "code": "const [photoFilter, setPhotoFilter] = useState('none');\\n\\nconst applyFilter = (filterType) => {{\\n  setPhotoFilter(filterType);\\n  // Apply filter logic here\\n}};",
          "explanation": "Managing photo filter state for an image editing application.",
          "context": "Photo editing app for photography enthusiasts"
        }}
      ],
      "practical_exercises": [
        {{
          "task": "Create a React component that tracks music playlist state with useState. Include functions to add/remove songs.",
          "solution_hint": "Use an array in useState and spread operator to update the playlist",
          "difficulty": "beginner"
        }},
        {{
          "task": "Build a game state manager that tracks player health, score, and level using multiple useState hooks.",
          "solution_hint": "Consider using separate state variables for different game aspects",
          "difficulty": "intermediate"
        }}
      ],
      "examples": [
        "Real-time gaming scoreboard",
        "Interactive photo gallery with filters",
        "Music player with playlist management"
      ]
    }}
  ]
}}

⚠️ YOU MUST FOLLOW THIS EXACT STRUCTURE FOR TECHNICAL LESSONS!''' if is_technical else ""}

INSTRUCTIONS:
1. Create a comprehensive, well-structured lesson with 5-7 substantial sections
2. Each section MUST have:
   - Clear, descriptive title
   - Rich, detailed content explanation (aim for 300-500 words per section)
   - Step-by-step breakdowns where appropriate
   - 3-4 practical examples that DIRECTLY connect to their interests ({interests_str}) and hobbies ({hobbies_str})
   - Real-world applications from software development industry

{f'''🔥 ADDITIONAL REQUIREMENTS FOR TECHNICAL LESSONS:
   - 2-4 COMPLETE, WORKING code examples per section with proper syntax highlighting
   - Each code example must have: language, code, explanation, and context (relating to student interests)
   - Code examples should be PRACTICAL and IMMEDIATELY USABLE
   - Include 2-3 hands-on exercises per section with solution hints
   - Exercise difficulty progression: "beginner", "intermediate", "advanced"
   - All code must be modern, using current best practices and latest syntax
   - Examples should demonstrate real-world industry patterns
   - Include error handling, edge cases, and debugging tips
   - Connect each code example to student's interests: {interests_str} and hobbies: {hobbies_str}
   - For React topics: functional components, hooks, modern JSX patterns
   - For JavaScript: ES6+, async/await, modern APIs
   - For any framework: current version syntax and best practices''' if is_technical else ""}

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

{f'''🎯 CODE QUALITY REQUIREMENTS (TECHNICAL LESSONS):
- All code must be production-ready and follow industry standards
- Include comments explaining complex logic
- Use meaningful variable names and clear function signatures
- Provide context about when and why to use each pattern
- Show both basic and advanced usage patterns
- Include performance considerations where relevant
- Demonstrate testing approaches for the code examples
- Connect to popular tools and libraries in the ecosystem

🚨 CRITICAL TECHNICAL VALIDATION CHECKLIST:
✅ Each section MUST have at least 2 code_examples with complete working code
✅ Each code example MUST have language, code, explanation, and context fields
✅ Each section MUST have at least 2 practical_exercises with task, solution_hint, and difficulty
✅ Code examples MUST be directly related to student interests: {interests_str}
✅ All code MUST use modern syntax (ES6+, React Hooks, current best practices)
✅ Examples MUST combine technical concepts with student hobbies: {hobbies_str}

FAILURE TO INCLUDE PROPER CODE EXAMPLES WILL RESULT IN LESSON REJECTION!''' if is_technical else ""}

{f'RESPONSE FORMAT: Return valid JSON matching the technical lesson schema shown above. Each section MUST contain both code_examples and practical_exercises arrays.' if is_technical else 'Generate a rich, personalized lesson in the specified JSON format with sections array.'}
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