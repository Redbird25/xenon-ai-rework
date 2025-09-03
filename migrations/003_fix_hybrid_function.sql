-- Fix hybrid_search function return types and null course filter handling

CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector,
    query_text text,
    course_id_filter uuid,
    similarity_threshold double precision DEFAULT 0.7,
    limit_results int DEFAULT 10
)
RETURNS TABLE(
    id integer,
    chunk_text text,
    source_ref text,
    similarity double precision,
    rank double precision
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_search AS (
        SELECT 
            lc.id,
            lc.chunk_text,
            lc.source_ref,
            1 - (lc.embedding <=> query_embedding) AS similarity
        FROM lesson_chunks lc
        WHERE lc.embedding IS NOT NULL
          AND (course_id_filter IS NULL OR lc.course_id = course_id_filter)
          AND 1 - (lc.embedding <=> query_embedding) > similarity_threshold
        ORDER BY similarity DESC
        LIMIT limit_results * 2
    ),
    text_search AS (
        SELECT 
            lc.id,
            lc.chunk_text,
            lc.source_ref,
            CAST(ts_rank(to_tsvector('english', lc.chunk_text), plainto_tsquery('english', query_text)) AS double precision) AS rank
        FROM lesson_chunks lc
        WHERE (course_id_filter IS NULL OR lc.course_id = course_id_filter)
          AND to_tsvector('english', lc.chunk_text) @@ plainto_tsquery('english', query_text)
        ORDER BY rank DESC
        LIMIT limit_results * 2
    )
    SELECT DISTINCT ON (COALESCE(v.id, t.id))
        COALESCE(v.id, t.id) AS id,
        COALESCE(v.chunk_text, t.chunk_text) AS chunk_text,
        COALESCE(v.source_ref, t.source_ref) AS source_ref,
        COALESCE(v.similarity, 0)::double precision AS similarity,
        COALESCE(t.rank, 0)::double precision AS rank
    FROM vector_search v
    FULL OUTER JOIN text_search t ON v.id = t.id
    ORDER BY COALESCE(v.id, t.id), (COALESCE(v.similarity, 0) + COALESCE(t.rank, 0)) DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;


