CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS lesson_chunks (
  id SERIAL PRIMARY KEY,
  lesson_id UUID NOT NULL,
  chunk_text TEXT NOT NULL,
  embedding VECTOR(1536),
  source_ref TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
ON lesson_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
