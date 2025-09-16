"""
Quiz generation and evaluation utilities
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import uuid
import math
from difflib import SequenceMatcher
import asyncio
import re

from app.db import async_session
from app.models import LessonChunk
from app.core.llm import get_llm_provider
from app.core.embeddings import get_embedding_service
from app.core.logging import get_logger, metrics_logger
from app.core.cache import get_cache
from app.core.llm import get_llm_provider
from app.core.vector_search import get_search_engine
from app.utils.language import choose_language, normalize_language_code


logger = get_logger(__name__)


ALLOWED_TYPES = {"open", "short_answer", "mcq_single", "mcq_multi"}


# Very small multilingual stopword list (en + uz + ru basics)
EN_STOPWORDS = {
    "the","and","or","a","an","of","to","in","on","for","with","by",
    "is","are","was","were","be","as","at","that","this","it","from","into",
    "about","over","under","between","within",
}
UZ_STOPWORDS = {
    "va","yoki","bu","shu","o'sha","ham","lekin","uchun","bilan","siz","biz","ular",
    "nima","qachon","qanday","qayerda","qaysi","bir","birinchi","ikkinchi","uchinchi","haqida",
}
RU_STOPWORDS = {
    "\u0438","\u043d\u043e","\u044d\u0442\u043e","\u043a\u0430\u043a","\u0447\u0442\u043e","\u044d\u0442\u0430","\u044d\u0442\u0438","\u044d\u0442\u043e\u0442",
    "\u0442\u0430\u043c","\u0437\u0434\u0435\u0441\u044c","\u0435\u0441\u0442\u044c","\u0434\u043b\u044f","\u0447\u0442\u043e\u0431\u044b","\u0438\u043b\u0438","\u0435\u0441\u043b\u0438","\u043a\u043e\u0433\u0434\u0430",
    "\u0442\u0430\u043a","\u0436\u0435","\u0447\u0435\u043c","\u0433\u0434\u0435","\u043a\u0442\u043e","\u043e\u043d\u0438","\u043c\u044b","\u0432\u044b",
    "\u043e\u043d","\u043e\u043d\u0430","\u043e\u043d\u043e","\u0438\u0445","\u0435\u0433\u043e","\u0435\u0435","\u0441\u0430\u043c","\u0441\u0430\u043c\u0430",
    "\u0441\u0430\u043c\u043e","\u0442\u043e\u0436\u0435","\u043e\u0447\u0435\u043d\u044c","\u043f\u043e\u0441\u043b\u0435","\u043f\u0435\u0440\u0435\u0434","\u043f\u0440\u0438","\u043d\u0430\u0434","\u043f\u043e\u0434",
    "\u043b\u0438\u0448\u044c","\u0440\u0430\u0437","\u0443\u0436\u0435","\u0435\u0449\u0451","\u0431\u044b","\u0431\u044b\u043b","\u0431\u044b\u043b\u0430","\u0431\u044b\u043b\u0438",
    "\u0431\u0443\u0434\u0435\u0442","\u0431\u0443\u0434\u0443\u0442","\u043a\u043e\u0442\u043e\u0440\u044b\u0439","\u043a\u0430\u043a\u043e\u0439","\u0442\u0430\u043a\u043e\u0439","\u044d\u0442\u043e\u0433\u043e","\u044d\u0442\u043e\u0439","\u044d\u0442\u043e\u043c",
    "\u044d\u0442\u0438\u0445",
}
STOPWORDS = EN_STOPWORDS | UZ_STOPWORDS | RU_STOPWORDS
NORMALIZE_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+", re.UNICODE)


def _extract_keywords(text: str, language: Optional[str] = None, max_terms: int = 20) -> List[str]:
    """Extract simple keyword set from text, removing stopwords and short tokens."""
    if not text:
        return []
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    kws = []
    seen = set()
    for tok in tokens:
        if tok.isdigit() or len(tok) < 3:
            continue
        if tok in STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        kws.append(tok)
        if len(kws) >= max_terms:
            break
    return kws


async def _fetch_chunks_texts(chunk_ids: List[int]) -> List[Dict[str, Any]]:
    """Load chunks from DB by ids with text and language metadata."""
    if not chunk_ids:
        return []
    async with async_session() as session:
        rows = []
        for cid in chunk_ids:
            ch = await session.get(LessonChunk, cid)
            if ch:
                lang = None
                if ch.meta and isinstance(ch.meta, dict):
                    lang = ch.meta.get("language")
                rows.append({
                    "id": ch.id,
                    "text": ch.chunk_text or "",
                    "language": lang or "en"
                })
        return rows


def _majority_language(chunks: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for c in chunks:
        lang = (c.get("language") or "en").lower()
        counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return "en"
    return max(counts.items(), key=lambda x: x[1])[0]


class QuizGenerator:
    """Generate quizzes from provided chunk ids and user preferences."""

    def __init__(self):
        self.llm = get_llm_provider()

    async def generate_quiz(
        self,
        chunk_ids: List[int],
        user_pref: Dict[str, Any],
        question_count: int = 10,
        open_ratio: float = 0.4,
        mcq_multi_allowed: bool = True,
        language_override: Optional[str] = None,
        topic_context: Optional[str] = None,
        topic_title: Optional[str] = None,
        topic_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Load chunks
        chunks = await _fetch_chunks_texts(chunk_ids)
        # Determine language using topic metadata and sampled content
        chunk_language_votes: list[str] = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                code = normalize_language_code(chunk.get("language"))
                if code:
                    chunk_language_votes.append(code)

        detection_samples: list[str] = []
        if topic_title:
            detection_samples.append(topic_title)
        if topic_description:
            detection_samples.append(topic_description)
        if topic_context:
            detection_samples.append(topic_context[:1200])
        for chunk in chunks[:5]:
            if isinstance(chunk, dict):
                snippet = (chunk.get("text") or "")[:400]
                if snippet and snippet.strip():
                    detection_samples.append(snippet)

        if language_override:
            language = normalize_language_code(language_override) or "en"
        else:
            language = choose_language(
                detection_samples,
                fallback="en",
                votes=chunk_language_votes,
                min_confidence=0.6,
            ) or "en"

        # Build compact context (truncate if too long)
        joined = "\n\n".join((c["text"] or "").strip() for c in chunks)
        # If no chunk content available, fall back to topic/title+description context
        if (not joined or not joined.strip()) and topic_context:
            joined = topic_context.strip()
        if len(joined) > 18000:
            joined = joined[:18000] + "..."

        # Compute target counts
        open_target = max(0, min(question_count, round(question_count * open_ratio)))
        closed_target = max(0, question_count - open_target)

        # JSON schema expected
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": question_count,
                    "maxItems": question_count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string"},
                            "prompt": {"type": "string"},
                            "options": {
                                "type": "array",
                                "items": {"type": "object", "properties": {
                                    "id": {"type": "string"},
                                    "text": {"type": "string"}
                                }, "required": ["id","text"]}
                            },
                            "correct_option_ids": {"type": "array", "items": {"type": "string"}},
                            "acceptable_answers": {"type": "array", "items": {"type": "string"}},
                            "acceptable_keywords": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                            "difficulty": {"type": "string"},
                            "explanation": {"type": "string"}
                        },
                        "required": ["id","type","prompt"]
                    }
                }
            },
            "required": ["questions"]
        }

        # Prompts
        # Language/script guidance
        script_hint = ""
        lang_lower = (language or "").lower()
        if lang_lower.startswith("uz-latn") or lang_lower == "uz":
            # Default Uzbek to Latin unless explicitly Cyrl
            script_hint = "Use Uzbek in Latin script (O?zbek lotin alifbosi). Do not use Cyrillic."
        elif lang_lower.startswith("uz-cyrl"):
            script_hint = "Use Uzbek in Cyrillic script (????? ?????? ????????)."
        else:
            # Do not add Uzbek-specific hints for non-Uzbek languages (e.g., 'ru')
            script_hint = script_hint

        system_prompt = (
            "You are an expert educational assessment designer. "
            "Create high-quality, unambiguous quiz questions strictly grounded in the provided topic and content. "
            "Allowed types: open, short_answer, mcq_single, mcq_multi. "
            + (script_hint or "")
        )

        # Topic anchors
        topic_anchor = ""
        if topic_title:
            topic_anchor += f"Topic title: {topic_title}\n"
        if topic_description:
            topic_anchor += f"Topic description: {topic_description}\n"

        user_instructions = (
            f"Language: {language}\n"
            f"{topic_anchor}"
            f"Total questions: {question_count}. Open target: {open_target}. Closed target: {closed_target}.\n"
            f"MCQ multi-select allowed: {str(mcq_multi_allowed).lower()}.\n"
            f"Personalization (style only): interests={user_pref.get('interests', [])}, "
            f"hobbies={user_pref.get('hobbies', [])}, learning_style={user_pref.get('learning_style','TEXT')}\n\n"
            "Rules (HARD CONSTRAINTS):\n"
            "- Do NOT create questions about the user's interests, hobbies, or learning style.\n"
            "- Every question must be about the topic and its concepts.\n"
            "- For open and short_answer include 3-6 acceptable_answers and 2-3 keyword sets (acceptable_keywords).\n"
            "- For short_answer aim for concise factual responses (single term/phrase).\n"
            "- For mcq_single provide 1 correct option; for mcq_multi provide 2 correct options when justified.\n"
            "- Options must be plausible and mutually exclusive; avoid 'All of the above'.\n"
            "- Avoid duplicates; vary difficulty (easy/medium/hard).\n"
            "- Ground all prompts strictly in the topic/content; avoid external facts.\n"
            "- Keep prompts concise and clear.\n\n"
            "Return only valid JSON matching the schema."
        )

        content_prompt = (
            "Source content (summarize across this text):\n\n" + joined
        )

        # Ask model
        raw_questions = await self.llm.generate_json(
            prompt=user_instructions + "\n\n" + content_prompt,
            schema=schema,
            system_prompt=system_prompt
        )

        questions = raw_questions.get("questions", []) if isinstance(raw_questions, dict) else []

        # Normalize
        normalized: List[Dict[str, Any]] = []
        used_ids: set[str] = set()
        for idx, q in enumerate(questions, start=1):
            if not isinstance(q, dict):
                continue
            qtype = str(q.get("type", "")).strip().lower()
            if qtype not in ALLOWED_TYPES:
                continue
            qid = str(q.get("id") or f"q{idx}")
            if qid in used_ids:
                qid = f"q{idx}-{uuid.uuid4().hex[:6]}"
            used_ids.add(qid)

            prompt = str(q.get("prompt") or "").strip()
            if not prompt:
                continue

            item: Dict[str, Any] = {
                "id": qid,
                "type": qtype,
                "prompt": prompt,
                "source_chunk_ids": list(chunk_ids),
                "difficulty": (str(q.get("difficulty") or "").lower() or "medium"),
                "explanation": (q.get("explanation") or "")
            }

            if qtype in {"open", "short_answer"}:
                acc = q.get("acceptable_answers") or []
                kw = q.get("acceptable_keywords") or []
                # Ensure minimum variants
                if isinstance(acc, list):
                    acc = [str(a).strip() for a in acc if str(a).strip()]
                else:
                    acc = []
                if len(acc) < 2:
                    acc.append(prompt)  # naive fallback
                if isinstance(kw, list):
                    kw_norm = []
                    for arr in kw:
                        if isinstance(arr, list):
                            kw_norm.append([str(x).strip() for x in arr if str(x).strip()])
                    kw = [k for k in kw_norm if k]
                else:
                    kw = []
                item["acceptable_answers"] = acc[:6]
                item["acceptable_keywords"] = kw[:3]

            else:  # MCQ
                opts = q.get("options") or []
                if not isinstance(opts, list):
                    continue
                options: List[Dict[str, str]] = []
                used_o_ids: set[str] = set()
                for oi, opt in enumerate(opts, start=1):
                    if not isinstance(opt, dict):
                        continue
                    oid = str(opt.get("id") or chr(ord('a') + oi - 1))
                    if oid in used_o_ids:
                        oid = f"{oid}{oi}"
                    used_o_ids.add(oid)
                    text = str(opt.get("text") or "").strip()
                    if not text:
                        continue
                    options.append({"id": oid, "text": text})
                if len(options) < 2:
                    continue
                correct = q.get("correct_option_ids") or []
                correct = [str(c) for c in correct if str(c)]
                if qtype == "mcq_single":
                    if len(correct) != 1:
                        # Try to clamp to first if provided, else skip
                        correct = correct[:1]
                        if not correct:
                            continue
                elif qtype == "mcq_multi":
                    if not correct or len(correct) < 2:
                        # require at least 2 correct for multi
                        continue
                    if not mcq_multi_allowed:
                        # downgrade to single by picking first correct
                        qtype = "mcq_single"
                        correct = correct[:1]
                item["type"] = qtype
                item["options"] = options
                item["correct_option_ids"] = correct

            normalized.append(item)

        # Balance counts to requested total and ratio
        def _split_by_type(items: List[Dict[str, Any]]):
            open_items = [x for x in items if x["type"] in {"open","short_answer"}]
            closed_items = [x for x in items if x["type"] in {"mcq_single","mcq_multi"}]
            return open_items, closed_items

        open_items, closed_items = _split_by_type(normalized)
        # Select up to targets first
        open_sel = open_items[:open_target]
        closed_sel = closed_items[:closed_target]
        selected = open_sel + closed_sel

        # If we have less than requested, attempt a top-up pass
        if len(selected) < question_count:
            remaining = question_count - len(selected)
            need_open = max(0, open_target - len(open_sel))
            need_closed = max(0, closed_target - len(closed_sel))

            # Try to pull additional from existing overflow first (to avoid extra LLM calls)
            extra_from_existing: List[Dict[str, Any]] = []
            extra_from_existing.extend(open_items[len(open_sel):])
            extra_from_existing.extend(closed_items[len(closed_sel):])

            # Deduplicate by prompt
            used_prompts = {q["prompt"].strip().lower() for q in selected}
            for q in extra_from_existing:
                if len(selected) >= question_count:
                    break
                p = str(q.get("prompt") or "").strip().lower()
                if not p or p in used_prompts:
                    continue
                selected.append(q)
                used_prompts.add(p)

            # If still short, ask LLM to generate exactly the remaining number
            if len(selected) < question_count:
                to_add = question_count - len(selected)
                add_open = min(need_open, to_add)
                add_closed = min(need_closed, max(0, to_add - add_open))
                # Build a constrained request
                add_schema: Dict[str, Any] = {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "minItems": to_add,
                            "maxItems": to_add,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {"type": "string"},
                                    "prompt": {"type": "string"},
                                    "options": {
                                        "type": "array",
                                        "items": {"type": "object", "properties": {
                                            "id": {"type": "string"},
                                            "text": {"type": "string"}
                                        }, "required": ["id","text"]}
                                    },
                                    "correct_option_ids": {"type": "array", "items": {"type": "string"}},
                                    "acceptable_answers": {"type": "array", "items": {"type": "string"}},
                                    "acceptable_keywords": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                                    "difficulty": {"type": "string"},
                                    "explanation": {"type": "string"}
                                },
                                "required": ["id","type","prompt"]
                            }
                        }
                    },
                    "required": ["questions"]
                }

                existing_prompts_str = "\n".join(f"- {pr}" for pr in used_prompts if pr)
                add_instructions = (
                    f"Language: {language}\n"
                    f"Add exactly {to_add} more questions strictly about the topic.\n"
                    f"Target additional count: open={add_open}, closed={add_closed} (use mcq_single/mcq_multi for closed).\n"
                    "Avoid any duplicates with this list (case-insensitive):\n"
                    f"{existing_prompts_str}\n\n"
                    "Return only valid JSON matching the schema."
                )
                add_content_prompt = (
                    "Source content (for context):\n\n" + joined
                )
                try:
                    add_raw = await self.llm.generate_json(
                        prompt=add_instructions + "\n\n" + add_content_prompt,
                        schema=add_schema,
                        system_prompt=system_prompt
                    )
                    add_qs = add_raw.get("questions", []) if isinstance(add_raw, dict) else []
                    # Normalize and append
                    for idx, q in enumerate(add_qs, start=1):
                        if not isinstance(q, dict):
                            continue
                        qtype = str(q.get("type", "")).strip().lower()
                        if qtype not in ALLOWED_TYPES:
                            continue
                        prompt_txt = str(q.get("prompt") or "").strip()
                        if not prompt_txt:
                            continue
                        if prompt_txt.strip().lower() in used_prompts:
                            continue
                        # Reuse normalization logic (inline minimal)
                        item: Dict[str, Any] = {
                            "id": str(q.get("id") or f"add{idx}"),
                            "type": qtype,
                            "prompt": prompt_txt,
                            "source_chunk_ids": list(chunk_ids),
                            "difficulty": (str(q.get("difficulty") or "").lower() or "medium"),
                            "explanation": (q.get("explanation") or "")
                        }
                        if qtype in {"open","short_answer"}:
                            acc = q.get("acceptable_answers") or []
                            acc = [str(a).strip() for a in acc if str(a).strip()]
                            if len(acc) < 2:
                                acc.append(prompt_txt)
                            kw = q.get("acceptable_keywords") or []
                            kw_norm = []
                            if isinstance(kw, list):
                                for arr in kw:
                                    if isinstance(arr, list):
                                        kw_norm.append([str(x).strip() for x in arr if str(x).strip()])
                            item["acceptable_answers"] = acc[:6]
                            item["acceptable_keywords"] = [k for k in kw_norm if k][:3]
                        else:
                            opts = q.get("options") or []
                            if not isinstance(opts, list):
                                continue
                            options: List[Dict[str, str]] = []
                            used_o_ids: set[str] = set()
                            for oi, opt in enumerate(opts, start=1):
                                if not isinstance(opt, dict):
                                    continue
                                oid = str(opt.get("id") or chr(ord('a') + oi - 1))
                                if oid in used_o_ids:
                                    oid = f"{oid}{oi}"
                                used_o_ids.add(oid)
                                text = str(opt.get("text") or "").strip()
                                if not text:
                                    continue
                                options.append({"id": oid, "text": text})
                            if len(options) < 2:
                                continue
                            correct = q.get("correct_option_ids") or []
                            correct = [str(c) for c in correct if str(c)]
                            if qtype == "mcq_single":
                                if len(correct) != 1:
                                    correct = correct[:1]
                                    if not correct:
                                        continue
                            elif qtype == "mcq_multi":
                                if not correct or len(correct) < 2:
                                    continue
                            item["options"] = options
                            item["correct_option_ids"] = correct

                        selected.append(item)
                        used_prompts.add(prompt_txt.strip().lower())
                        if len(selected) >= question_count:
                            break
                except Exception:
                    pass

        quiz = {
            "quiz_id": str(uuid.uuid4()),
            "language": language,
            "questions": selected,
            "meta": {
                "requested_total": question_count,
                "requested_open": open_target,
                "requested_closed": closed_target,
                "generated_total": len(selected)
            }
        }
        return quiz


async def select_chunk_ids_for_topic(
    title: str,
    description: Optional[str],
    language: Optional[str],
    top_k: int = 40
) -> List[int]:
    """Select most relevant chunk ids using stricter filtering (hybrid + keyword checks)."""
    query = (title or "").strip()
    if description and description.strip():
        query = f"{query}\n\n{description.strip()}"

    filters = {"language": language} if language else None
    engine = get_search_engine()
    # Pull extra candidates; we will filter strictly afterwards
    results = await engine.search(
        query=query,
        course_id=None,
        top_k=max(top_k, 20) * 2,
        similarity_threshold=0.75,
        filters=filters,
        use_hybrid=True,
        track_query=False
    )
    # Keyword-based filter to reduce off-topic
    kws = _extract_keywords(query, language)
    min_hits = 2 if len(kws) >= 4 else 1

    # Unique preserve order with score and per-source cap
    seen: set[int] = set()
    per_source: Dict[str, int] = {}
    max_per_source = 6
    selected: List[int] = []
    for r in results:
        # Combined score threshold
        if getattr(r, "combined_score", 0.0) < 0.70:
            continue
        # Keyword hits in content
        content_l = (r.content or "").lower()
        if kws:
            hits = sum(1 for k in kws if k in content_l)
            if hits < min_hits:
                continue
        # Per-source cap
        src = (r.source_ref or "").strip() or "__none__"
        cnt = per_source.get(src, 0)
        if cnt >= max_per_source:
            continue
        per_source[src] = cnt + 1

        cid = int(r.chunk_id)
        if cid not in seen:
            seen.add(cid)
            selected.append(cid)
        if len(selected) >= top_k:
            break
    return selected


def sanitize_quiz_for_delivery(quiz: Dict[str, Any]) -> Dict[str, Any]:
    """Return a minimal view for Core: only questions and options (if any).
    Keeps quiz_id for correlation; strips answers/keys/metadata.
    """
    qid = quiz.get("quiz_id") or quiz.get("quizId")
    out: Dict[str, Any] = {
        "quiz_id": qid,
        "language": quiz.get("language"),
        "questions": []
    }
    questions = quiz.get("questions", []) or []
    for q in questions:
        if not isinstance(q, dict):
            continue
        item: Dict[str, Any] = {
            "id": q.get("id"),
            "type": q.get("type"),
            "prompt": q.get("prompt"),
        }
        opts = q.get("options")
        if isinstance(opts, list) and opts:
            clean_opts = []
            for opt in opts:
                if isinstance(opt, dict):
                    oid = opt.get("id")
                    txt = opt.get("text")
                    if oid is not None and txt:
                        clean_opts.append({"id": oid, "text": txt})
            if clean_opts:
                item["options"] = clean_opts
        out["questions"].append(item)
    return out


class AnswerEvaluator:
    """Evaluate user answers against quiz content using embeddings and heuristics."""

    def __init__(self):
        self.embed = get_embedding_service()
        self.llm = get_llm_provider()
        self.cache = get_cache()

    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        return await self.embed.provider.embed_documents(texts)

    async def _best_similarity(self, user: str, candidates: List[str]) -> float:
        if not candidates:
            return 0.0
        vec_u = await self.embed.provider.embed_query(user)
        cand_vecs = await self._embed_texts(candidates)
        best = 0.0
        for v in cand_vecs:
            s = self.embed.compute_similarity(vec_u, v)
            if s > best:
                best = s
        return float(best)


    @staticmethod
    def _normalize_text(value: str) -> str:
        if not value:
            return ""
        lowered = value.lower()
        cleaned = NORMALIZE_PUNCT_RE.sub(" ", lowered)
        return MULTISPACE_RE.sub(" ", cleaned).strip()

    @staticmethod
    def _tokenize(normalized_text: str) -> List[str]:
        if not normalized_text:
            return []
        return [tok for tok in normalized_text.split() if tok and tok not in STOPWORDS]

    @staticmethod
    def _prompt_key(text: str) -> str:
        normalized = AnswerEvaluator._normalize_text(text)
        return normalized.replace(" ", "")

    def _normalize_option_answer(self, raw_answer: Any, options: List[Dict[str, Any]]) -> Optional[str]:
        if raw_answer is None:
            return None
        raw_str = str(raw_answer).strip()
        if not raw_str:
            return None
        lowered = raw_str.lower()
        id_lookup: dict[str, str] = {}
        text_lookup: dict[str, str] = {}
        letter_lookup: dict[str, str] = {}
        for idx, option in enumerate(options or []):
            if not isinstance(option, dict):
                continue
            opt_id = str(option.get('id') or '').strip()
            opt_text = option.get('text') or ''
            if opt_id:
                id_lookup[opt_id.lower()] = opt_id
            norm_text = self._normalize_text(opt_text)
            if norm_text:
                text_lookup[norm_text] = opt_id or norm_text
            letter = chr(ord('a') + idx)
            if opt_id:
                letter_lookup[letter] = opt_id
                letter_lookup[letter.upper()] = opt_id
        if lowered in id_lookup:
            return id_lookup[lowered]
        if lowered in letter_lookup:
            return letter_lookup[lowered]
        normalized_text = self._normalize_text(raw_str)
        if normalized_text in text_lookup:
            return text_lookup[normalized_text]
        return raw_str

    def _token_overlap_score(self, user_tokens: List[str], candidate_tokens: List[str]) -> float:
        if not user_tokens or not candidate_tokens:
            return 0.0
        user_set = set(user_tokens)
        candidate_set = set(candidate_tokens)
        if not candidate_set:
            return 0.0
        jaccard = len(user_set & candidate_set) / max(1, len(user_set | candidate_set))
        coverage = len(user_set & candidate_set) / max(1, len(candidate_set))
        return float(max(jaccard, coverage))

    def _best_literal_similarity(self, user_normalized: str, candidate_normalized: List[str]) -> float:
        if not user_normalized or not candidate_normalized:
            return 0.0
        best = 0.0
        for candidate in candidate_normalized:
            if not candidate:
                continue
            if user_normalized == candidate:
                return 1.0
            if user_normalized in candidate or candidate in user_normalized:
                coverage = min(len(user_normalized), len(candidate)) / max(len(user_normalized), len(candidate))
                if coverage > best:
                    best = coverage
            ratio = SequenceMatcher(None, user_normalized, candidate).ratio()
            if ratio > best:
                best = ratio
        return float(best)
    @staticmethod
    def _keyword_overlap(user: str, keyword_sets: List[List[str]]) -> float:
        if not keyword_sets:
            return 0.0
        u = user.lower()
        best = 0.0
        for ks in keyword_sets:
            if not ks:
                continue
            hits = sum(1 for k in ks if k.lower() in u)
            ratio = hits / max(1, len(ks))
            if ratio > best:
                best = ratio
        return float(best)

    async def _context_support(self, user: str, chunk_ids: List[int]) -> float:
        # Simple embedding similarity between user answer and chunk texts (restricted to provided ids)
        chunks = await _fetch_chunks_texts(chunk_ids)
        texts = [c["text"] for c in chunks if c.get("text")]
        if not texts:
            return 0.0
        uvec = await self.embed.provider.embed_query(user)
        cvecs = await self._embed_texts(texts)
        best = 0.0
        for v in cvecs:
            s = self.embed.compute_similarity(uvec, v)
            if s > best:
                best = s
        return float(best)

    async def evaluate(self, content: Dict[str, Any], answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        q_by_id = {q["id"]: q for q in content.get("questions", [])}
        details: List[Dict[str, Any]] = []
        total = 0.0
        count = 0

        # Pre-normalize answer map
        a_map: Dict[str, Any] = {}
        for a in answers:
            qid = str(a.get("question_id"))
            a_map[qid] = a.get("answer")

        for qid, q in q_by_id.items():
            qtype = q.get("type")
            user_ans = a_map.get(qid)
            score = 0.0
            verdict = "incorrect"
            explanation = None

            if qtype in {"mcq_single", "mcq_multi"}:
                expected = set(str(x) for x in (q.get("correct_option_ids") or []))
                options = q.get("options") or []
                normalized_answers: list[str] = []
                answers_iter = user_ans if isinstance(user_ans, list) else [user_ans]
                for raw in answers_iter:
                    mapped = self._normalize_option_answer(raw, options)
                    if mapped is not None:
                        normalized_answers.append(str(mapped))
                ua_set = set(normalized_answers)
                if qtype == "mcq_single":
                    score = 1.0 if len(expected) == 1 and ua_set == expected else 0.0
                else:
                    inter = len(expected & ua_set)
                    union = len(expected | ua_set) if (expected or ua_set) else 1
                    j = inter / union
                    over = max(0, len(ua_set - expected))
                    score = max(0.0, j - 0.1 * over)
                    score = float(min(1.0, score))
                verdict = "correct" if score >= 0.995 else ("partial" if score >= 0.5 else "incorrect")
                explanation = "Checked selected options against correct keys."

            elif qtype in {"open", "short_answer"}:
                # Accept answers always as array (join to single string)
                if isinstance(user_ans, list):
                    ua = " ".join([str(x).strip() for x in user_ans if str(x).strip()])
                else:
                    ua = str(user_ans or "").strip()
                if not ua:
                    score = 0.0
                    verdict = "incorrect"
                    explanation = "No answer provided."
                else:
                    acc: List[str] = q.get("acceptable_answers") or []
                    kws: List[List[str]] = q.get("acceptable_keywords") or []
                    best_sim = await self._best_similarity(ua, acc) if acc else 0.0
                    kw_score = self._keyword_overlap(ua, kws)
                    ctx_score = await self._context_support(ua, q.get("source_chunk_ids") or [])
                    normalized_expected = [self._normalize_text(ans) for ans in acc]
                    normalized_user = self._normalize_text(ua)
                    literal_score = self._best_literal_similarity(normalized_user, normalized_expected)
                    user_tokens = self._tokenize(normalized_user)
                    token_score = 0.0
                    if normalized_expected:
                        token_scores = [
                            self._token_overlap_score(user_tokens, self._tokenize(ans))
                            for ans in normalized_expected
                        ]
                        if token_scores:
                            token_score = max(token_scores)
                    base_similarity = max(best_sim, literal_score)
                    if qtype == "short_answer":
                        score = (
                            0.5 * base_similarity
                            + 0.25 * token_score
                            + 0.15 * kw_score
                            + 0.10 * ctx_score
                        )
                    else:
                        score = (
                            0.45 * base_similarity
                            + 0.20 * token_score
                            + 0.20 * kw_score
                            + 0.15 * ctx_score
                        )
                    score = max(score, literal_score, token_score, best_sim)
                    score = float(min(1.0, max(0.0, score)))
                    verdict = "correct" if score >= 0.78 else ("partial" if score >= 0.55 else "incorrect")
                    explanation = (
                        f"emb={best_sim:.2f}, literal={literal_score:.2f}, "
                        f"tokens={token_score:.2f}, keywords={kw_score:.2f}, context={ctx_score:.2f}"
                    )
            details.append({
                "question_id": qid,
                "verdict": verdict,
                "score": round(score, 3),
                "explanation": explanation
            })
            total += score
            count += 1

        score_ratio = (total / count) if count else 0.0
        return {
            "quiz_id": content.get("quiz_id"),
            "score_percent": round(score_ratio, 4),
            "details": details
        }

    async def _expected_from_context(self, question: str, chunk_ids: List[int], language: Optional[str]) -> Dict[str, Any]:
        """Use LLM to derive acceptable answers and keyword sets from provided context."""
        chunks = await _fetch_chunks_texts(chunk_ids)
        joined = "\n\n".join((c["text"] or "").strip() for c in chunks if c.get("text"))
        if not joined:
            # No context - ask model to propose concise canonical answers from the question itself
            joined = ""
        if len(joined) > 16000:
            joined = joined[:16000] + "..."

        schema = {
            "type": "object",
            "properties": {
                "acceptable_answers": {"type": "array", "items": {"type": "string"}},
                "acceptable_keywords": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
            },
            "required": ["acceptable_answers", "acceptable_keywords"]
        }

        sys_prompt = (
            "You extract canonical answer variants and key term sets from the given context to grade a student's answer. "
            "Keep outputs concise and language-aware."
        )
        user_prompt = (
            f"Language: {language or ''}\n"
            f"Question: {question}\n\n"
            f"Context (may be empty):\n{joined}\n\n"
            "Return JSON with 3-6 acceptable_answers and 2-3 acceptable_keywords sets."
        )
        try:
            data = await self.llm.generate_json(user_prompt, schema, sys_prompt)
        except Exception:
            data = {"acceptable_answers": [], "acceptable_keywords": []}
        # Normalize
        acc = [str(x).strip() for x in (data.get("acceptable_answers") or []) if str(x).strip()]
        kw = []
        for arr in (data.get("acceptable_keywords") or []):
            if isinstance(arr, list):
                kw.append([str(x).strip() for x in arr if str(x).strip()])
        return {"acceptable_answers": acc[:6], "acceptable_keywords": kw[:3]}

    async def evaluate_minimal(
        self,
        items: List[Dict[str, Any]],
        language: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate when only question text and user answer are provided."""
        details: List[Dict[str, Any]] = []
        total = 0.0
        count = 0
        import uuid as _uuid
        quiz_id = str(_uuid.uuid4())

        import hashlib
        for idx, item in enumerate(items, start=1):
            qtext = str(item.get("question") or "").strip()
            uans = item.get("answer")
            if not qtext:
                continue
            # Cache key includes language and normalized question
            base = f"{(language or '').lower()}|{qtext.lower()}".encode("utf-8")
            qhash = hashlib.sha256(base).hexdigest()
            cache_key = f"quiz:spec:{qhash}"

            # Try cache first
            spec = await self.cache.get_json(cache_key)
            if spec and isinstance(spec, dict):
                expected = {
                    "acceptable_answers": spec.get("acceptable_answers", []),
                    "acceptable_keywords": spec.get("acceptable_keywords", [])
                }
                chunk_ids = [int(x) for x in spec.get("source_chunk_ids", []) if isinstance(x, (int, float, str))]
            else:
                # Retrieve context strictly by question (+ optional description)
                chunk_ids = await select_chunk_ids_for_topic(
                    title=qtext,
                    description=description,
                    language=language,
                    top_k=20
                )
                # Derive expected answers from context using LLM
                expected = await self._expected_from_context(qtext, chunk_ids, language)
                # Store to cache with TTL
                await self.cache.set_json(cache_key, {
                    "acceptable_answers": expected.get("acceptable_answers", []),
                    "acceptable_keywords": expected.get("acceptable_keywords", []),
                    "source_chunk_ids": chunk_ids,
                    "language": language or ""
                })
            qmock = {
                "id": f"q{idx}",
                "type": "open",
                "prompt": qtext,
                "acceptable_answers": expected.get("acceptable_answers", []),
                "acceptable_keywords": expected.get("acceptable_keywords", []),
                "source_chunk_ids": chunk_ids
            }
            # Reuse existing scoring for open questions
            result = await self.evaluate({"quiz_id": quiz_id, "questions": [qmock]}, [{"question_id": qmock["id"], "answer": uans}])
            # Pull the single detail
            if result.get("details"):
                details.append(result["details"][0])
                total += result["details"][0]["score"]
                count += 1

        score_ratio = (total / count) if count else 0.0
        return {"quiz_id": quiz_id, "score_percent": round(score_ratio, 4), "details": details}

    async def evaluate_by_lesson(self, lesson_material_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate answers using full quiz spec fetched by lesson_material_id from Redis cache."""
        cache = get_cache()
        key = f"quiz:lesson:{lesson_material_id}"
        spec = await cache.get_json(key)
        if not spec or not isinstance(spec, dict):
            # Fallback to minimal without context
            return await self.evaluate_minimal(items=items)

        questions = spec.get("questions", []) or []
        # Build index by normalized prompt
        def norm(s: str) -> str:
            return self._prompt_key(s)

        idx_by_prompt: Dict[str, Dict[str, Any]] = {}
        for q in questions:
            if isinstance(q, dict) and q.get("prompt"):
                idx_by_prompt[norm(str(q.get("prompt")))] = q

        # Build content and answers for the core evaluator
        content_questions: List[Dict[str, Any]] = []
        answers: List[Dict[str, Any]] = []
        import uuid as _uuid
        quiz_id = spec.get("quiz_id") or str(_uuid.uuid4())

        for i, item in enumerate(items, start=1):
            qtext = str(item.get("question") or "").strip()
            uans = item.get("answer")
            if not qtext:
                continue
            match = idx_by_prompt.get(norm(qtext))
            if match:
                qid = match.get("id") or f"q{i}"
                # Use the stored spec question as-is (contains all grading info)
                content_questions.append({**match, "id": qid})
                answers.append({"question_id": qid, "answer": uans})
            else:
                # No exact prompt match - fallback by creating minimal spec via context of stored quiz
                # Try to use union of all chunk ids from spec for some context support
                chunk_union = []
                try:
                    for q in questions:
                        if isinstance(q, dict):
                            chunk_union.extend([int(x) for x in (q.get("source_chunk_ids") or [])])
                except Exception:
                    pass
                expected = await self._expected_from_context(qtext, list(set(chunk_union)), spec.get("language"))
                qmock = {
                    "id": f"q{i}",
                    "type": "open",
                    "prompt": qtext,
                    "acceptable_answers": expected.get("acceptable_answers", []),
                    "acceptable_keywords": expected.get("acceptable_keywords", []),
                    "source_chunk_ids": list(set(chunk_union))
                }
                content_questions.append(qmock)
                answers.append({"question_id": qmock["id"], "answer": uans})

        result = await self.evaluate({"quiz_id": quiz_id, "questions": content_questions}, answers)
        return result

    async def evaluate_by_quiz_id(self, quiz_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate answers using full quiz spec fetched by quiz_id from Redis cache."""
        cache = get_cache()
        key = f"quiz:id:{quiz_id}"
        spec = await cache.get_json(key)
        if not spec or not isinstance(spec, dict):
            # Fallback minimal if no spec found
            return await self.evaluate_minimal(items=items)

        questions = spec.get("questions", []) or []
        language = spec.get("language")

        def norm(s: str) -> str:
            return self._prompt_key(s)

        idx_by_prompt: Dict[str, Dict[str, Any]] = {}
        for q in questions:
            if isinstance(q, dict) and q.get("prompt"):
                idx_by_prompt[norm(str(q.get("prompt")))] = q

        content_questions: List[Dict[str, Any]] = []
        answers: List[Dict[str, Any]] = []
        for i, item in enumerate(items, start=1):
            qtext = str(item.get("question") or "").strip()
            uans = item.get("answer")
            if not qtext:
                continue
            match = idx_by_prompt.get(norm(qtext))
            if match:
                qid = match.get("id") or f"q{i}"
                content_questions.append({**match, "id": qid})
                answers.append({"question_id": qid, "answer": uans})
            else:
                # Build expected from all chunks in this quiz spec
                chunk_union: List[int] = []
                try:
                    for q in questions:
                        if isinstance(q, dict):
                            chunk_union.extend([int(x) for x in (q.get("source_chunk_ids") or [])])
                except Exception:
                    pass
                expected = await self._expected_from_context(qtext, list(set(chunk_union)), language)
                qmock = {
                    "id": f"q{i}",
                    "type": "open",
                    "prompt": qtext,
                    "acceptable_answers": expected.get("acceptable_answers", []),
                    "acceptable_keywords": expected.get("acceptable_keywords", []),
                    "source_chunk_ids": list(set(chunk_union))
                }
                content_questions.append(qmock)
                answers.append({"question_id": qmock["id"], "answer": uans})

        return await self.evaluate({"quiz_id": quiz_id, "questions": content_questions}, answers)
