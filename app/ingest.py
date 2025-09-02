import google.generativeai as genai
from app.config import GEMINI_API_KEY
from app.db import async_session
from app.models import LessonChunk

genai.configure(api_key=GEMINI_API_KEY)

async def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for t in texts:
        resp = genai.embed_content(
            model="models/text-embedding-004",
            content=t
        )
        embeddings.append(resp["embedding"])
    return embeddings


async def ingest_text(course_id: str, text: str, source_ref: str | None = None):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]  # naive split
    embeddings = await embed_texts(chunks)

    async with async_session() as session:
        for chunk, emb in zip(chunks, embeddings):
            lc = LessonChunk(
                course_id=course_id,
                chunk_text=chunk,
                embedding=emb,
                source_ref=source_ref
            )
            session.add(lc)
        await session.commit()
