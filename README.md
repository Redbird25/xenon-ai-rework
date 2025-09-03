# AI Ingest Service 2.0

Modern document ingestion service with advanced RAG (Retrieval-Augmented Generation) capabilities, built with FastAPI, LangChain, and PostgreSQL with pgvector.

## Features

- **🚀 Modern Architecture**: Built with FastAPI, async/await, and best practices
- **📚 Multi-format Support**: PDF, HTML, Markdown, plain text
- **🧠 Advanced Chunking**: Multiple strategies (fixed, recursive, semantic, markdown-aware)
- **🔍 Hybrid Search**: Combines vector similarity and keyword search
- **🌐 Multi-provider Support**: OpenAI, Google Gemini, Anthropic, local models
- **📊 Monitoring**: Structured logging, Prometheus metrics, health checks
- **⚡ Performance**: Async processing, connection pooling, caching
- **🔄 Resilience**: Retry mechanisms, circuit breakers, rate limiting
- **📈 Analytics**: Search query tracking and feedback collection

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ with pgvector extension
- Redis (optional, for caching)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-ingest
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run database migrations:
```bash
alembic upgrade head
```

5. Start the service:
```bash
uvicorn app.main:app --reload --port 8001
```

## Environment Variables

```bash
# Environment
ENVIRONMENT=development  # development, staging, production
DEBUG=true

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/xenon_ai_db
DATABASE_POOL_SIZE=10

# API Keys (at least one required)
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# LLM Configuration
LLM_MODEL=gemini-1.5-flash
LLM_TEMPERATURE=0.0
EMBEDDING_MODEL=models/embedding-001

# Chunking
CHUNKING_STRATEGY=recursive  # fixed, recursive, semantic, markdown
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# External Services
CORE_CALLBACK_URL=http://core-service:8080/api/ingest/callback

# Optional
REDIS_URL=redis://localhost:6379/0
LOG_LEVEL=INFO
```

## API Endpoints

### Document Ingestion

```bash
# Ingest documents
curl -X POST http://localhost:8001/ai/ingest/resources \
  -H "Content-Type: application/json" \
  -d '{
    "course_id": "c56a4180-65aa-42ec-a945-5fd21dec0538",
    "title": "Introduction to Python",
    "description": "Learn Python programming basics",
    "resources": [
      "https://example.com/python-tutorial.pdf",
      "https://docs.python.org/3/tutorial/index.html"
    ],
    "lang": "en"
  }'
```

### Vector Search

```bash
# Search documents
curl -X POST http://localhost:8001/ai/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are Python decorators?",
    "course_id": "c56a4180-65aa-42ec-a945-5fd21dec0538",
    "top_k": 5,
    "use_hybrid": true
  }'

# Find similar chunks
curl -X GET http://localhost:8001/ai/search/similar/123?top_k=5

# Search by metadata
curl -X POST http://localhost:8001/ai/search/metadata \
  -H "Content-Type: application/json" \
  -d '{
    "language": "en",
    "document_type": "pdf"
  }'
```

### Health & Monitoring

```bash
# Health check
curl http://localhost:8001/health

# Readiness check (includes DB connectivity)
curl http://localhost:8001/health/ready

# Prometheus metrics
curl http://localhost:8001/metrics
```

## Architecture

### Core Components

1. **Document Loaders** (`app/core/document_loaders.py`)
   - Support for multiple file formats
   - Automatic format detection
   - Metadata extraction

2. **Chunking System** (`app/core/chunking.py`)
   - Multiple chunking strategies
   - Content-aware splitting
   - Metadata preservation

3. **LLM Integration** (`app/core/llm.py`)
   - Multi-provider support via LangChain
   - Document cleaning
   - Query enhancement
   - Course route generation

4. **Embedding Service** (`app/core/embeddings.py`)
   - Multiple embedding models
   - Batch processing
   - Dimension management

5. **Vector Search** (`app/core/vector_search.py`)
   - Hybrid search (vector + keyword)
   - Metadata filtering
   - Similarity search
   - Query tracking

6. **Retry & Resilience** (`app/core/retry.py`)
   - Circuit breakers
   - Exponential backoff
   - Rate limiting

## Database Schema

The service uses PostgreSQL with pgvector extension for vector operations:

- `lesson_chunks`: Document chunks with embeddings
- `documents`: Source document metadata
- `search_queries`: Search analytics
- `ingest_jobs`: Job tracking

## Docker Support

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Linting
ruff check app/

# Type checking
mypy app/

# Format code
black app/
```

## Monitoring & Observability

- **Structured Logging**: JSON logs with request tracking
- **Metrics**: Prometheus metrics for all operations
- **Tracing**: OpenTelemetry support (optional)
- **Health Checks**: Liveness and readiness probes

## Performance Tuning

1. **Database**: Adjust `DATABASE_POOL_SIZE` based on load
2. **Embeddings**: Tune `EMBEDDING_BATCH_SIZE` for throughput
3. **Chunking**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP`
4. **Search**: Configure `similarity_threshold` for precision/recall

## License

MIT License
