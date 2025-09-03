-- Migration to update schema for modern RAG capabilities

-- Update lesson_chunks table
ALTER TABLE lesson_chunks 
    ADD COLUMN IF NOT EXISTS document_id UUID,
    ADD COLUMN IF NOT EXISTS course_id UUID,
    ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now(),
    ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100),
    ADD COLUMN IF NOT EXISTS chunk_index INTEGER,
    ADD COLUMN IF NOT EXISTS chunk_size INTEGER,
    ADD COLUMN IF NOT EXISTS search_vector TEXT;

-- Update embedding column to support dynamic dimensions
ALTER TABLE lesson_chunks ALTER COLUMN embedding TYPE vector;

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_lesson_chunks_document_id ON lesson_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_lesson_chunks_course_id ON lesson_chunks(course_id);
CREATE INDEX IF NOT EXISTS idx_lesson_chunks_metadata ON lesson_chunks USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_lesson_chunks_search_vector ON lesson_chunks USING gin(to_tsvector('english', search_vector));

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id UUID NOT NULL,
    title VARCHAR(500),
    source_url TEXT,
    document_type VARCHAR(50),
    language VARCHAR(10),
    processing_status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    chunks_count INTEGER DEFAULT 0,
    content_hash VARCHAR(64) UNIQUE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_documents_course_id ON documents(course_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);

-- Create search_queries table for analytics
CREATE TABLE IF NOT EXISTS search_queries (
    id SERIAL PRIMARY KEY,
    course_id UUID,
    query_text TEXT NOT NULL,
    enhanced_query TEXT,
    query_embedding vector,
    search_params JSONB DEFAULT '{}',
    filters JSONB DEFAULT '{}',
    results_count INTEGER,
    top_score FLOAT,
    response_time_ms INTEGER,
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    user_feedback TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_search_queries_course_id ON search_queries(course_id);
CREATE INDEX IF NOT EXISTS idx_search_queries_created_at ON search_queries(created_at);

-- Create ingest_jobs table
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id UUID NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    job_type VARCHAR(50),
    total_items INTEGER,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,
    error_message TEXT,
    result_data JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_ingest_jobs_course_id ON ingest_jobs(course_id);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_status ON ingest_jobs(status);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_created_at ON ingest_jobs(created_at);

-- Update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_lesson_chunks_updated_at 
    BEFORE UPDATE ON lesson_chunks 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Add function for hybrid search
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector,
    query_text text,
    course_id_filter uuid,
    similarity_threshold float DEFAULT 0.7,
    limit_results int DEFAULT 10
)
RETURNS TABLE(
    id integer,
    chunk_text text,
    source_ref text,
    similarity float,
    rank float
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_search AS (
        SELECT 
            lc.id,
            lc.chunk_text,
            lc.source_ref,
            1 - (lc.embedding <=> query_embedding) as similarity
        FROM lesson_chunks lc
        WHERE lc.course_id = course_id_filter
            AND lc.embedding IS NOT NULL
            AND 1 - (lc.embedding <=> query_embedding) > similarity_threshold
        ORDER BY similarity DESC
        LIMIT limit_results * 2
    ),
    text_search AS (
        SELECT 
            lc.id,
            lc.chunk_text,
            lc.source_ref,
            ts_rank(to_tsvector('english', lc.chunk_text), plainto_tsquery('english', query_text)) as rank
        FROM lesson_chunks lc
        WHERE lc.course_id = course_id_filter
            AND to_tsvector('english', lc.chunk_text) @@ plainto_tsquery('english', query_text)
        ORDER BY rank DESC
        LIMIT limit_results * 2
    )
    SELECT DISTINCT ON (COALESCE(v.id, t.id))
        COALESCE(v.id, t.id) as id,
        COALESCE(v.chunk_text, t.chunk_text) as chunk_text,
        COALESCE(v.source_ref, t.source_ref) as source_ref,
        COALESCE(v.similarity, 0) as similarity,
        COALESCE(t.rank, 0) as rank
    FROM vector_search v
    FULL OUTER JOIN text_search t ON v.id = t.id
    ORDER BY COALESCE(v.id, t.id), (COALESCE(v.similarity, 0) + COALESCE(t.rank, 0)) DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;
