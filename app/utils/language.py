from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional
import re

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0


@dataclass
class LanguageGuess:
    """Represents a detected language code with confidence."""

    code: str
    confidence: float


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
_LATIN_RANGES = ((0x0041, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF))
_CYRILLIC_RANGES = ((0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))


def normalize_language_code(code: Optional[str]) -> Optional[str]:
    """Normalize language tags to a consistent casing (for example uz-Cyrl)."""
    if not code:
        return None
    cleaned = code.replace("_", "-").strip()
    if not cleaned:
        return None
    parts = cleaned.split("-")
    if not parts:
        return None
    base = parts[0].lower()
    script_parts = [p.title() for p in parts[1:]]
    normalized = "-".join([base, *script_parts]) if script_parts else base
    return normalized


def _char_in_ranges(ch: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    code_point = ord(ch)
    return any(start <= code_point <= end for start, end in ranges)


def _script_counts(text: str) -> tuple[int, int]:
    latin_count = 0
    cyrillic_count = 0
    for ch in text:
        if _char_in_ranges(ch, _CYRILLIC_RANGES):
            cyrillic_count += 1
        elif _char_in_ranges(ch, _LATIN_RANGES):
            latin_count += 1
    return latin_count, cyrillic_count


def _prepare_texts(texts: Iterable[str], max_chars: int = 4000) -> str:
    collected: list[str] = []
    total = 0
    for raw in texts:
        if not raw:
            continue
        cleaned = _URL_RE.sub(" ", str(raw))
        cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
        if not cleaned:
            continue
        if max_chars > 0 and total + len(cleaned) > max_chars:
            cleaned = cleaned[: max_chars - total]
        collected.append(cleaned)
        total += len(cleaned)
        if max_chars > 0 and total >= max_chars:
            break
    return " ".join(collected).strip()


def _refine_language(code: str, sample_text: str) -> str:
    normalized = normalize_language_code(code)
    if not normalized:
        return code
    base = normalized.split("-")[0]
    if base == "uz":
        latin, cyrillic = _script_counts(sample_text)
        if cyrillic and cyrillic >= latin:
            return "uz-Cyrl"
        if latin:
            return "uz-Latn"
        return "uz"
    return normalized


def detect_language(texts: Iterable[str], fallback: Optional[str] = None, *, min_confidence: float = 0.55) -> Optional[LanguageGuess]:
    """Detect language from a collection of text fragments."""
    merged = _prepare_texts(texts)
    if not merged:
        fallback_code = normalize_language_code(fallback)
        return LanguageGuess(fallback_code, 0.0) if fallback_code else None
    try:
        candidates = detect_langs(merged)
    except LangDetectException:
        fallback_code = normalize_language_code(fallback)
        return LanguageGuess(fallback_code, 0.0) if fallback_code else None
    if not candidates:
        fallback_code = normalize_language_code(fallback)
        return LanguageGuess(fallback_code, 0.0) if fallback_code else None
    best_guess: Optional[LanguageGuess] = None
    for candidate in candidates:
        refined = _refine_language(candidate.lang, merged)
        guess = LanguageGuess(refined, candidate.prob)
        if candidate.prob >= min_confidence:
            return guess
        if best_guess is None or candidate.prob > best_guess.confidence:
            best_guess = guess
    if best_guess:
        return best_guess
    fallback_code = normalize_language_code(fallback)
    return LanguageGuess(fallback_code, 0.0) if fallback_code else None


def choose_language(texts: Iterable[str], *, fallback: Optional[str] = "en", preferred: Optional[str] = None, votes: Optional[Iterable[str]] = None, min_confidence: float = 0.55) -> Optional[str]:
    """Choose the best language using detection, optional preference, and votes."""
    preferred_code = normalize_language_code(preferred)
    vote_codes = [normalize_language_code(v) for v in (votes or []) if v]
    detected = detect_language(texts, fallback=None, min_confidence=min_confidence)
    if detected and detected.confidence >= min_confidence:
        return detected.code
    if vote_codes:
        counts = Counter(vote_codes)
        vote_code, _ = counts.most_common(1)[0]
        if detected is None or detected.confidence < min_confidence:
            return vote_code
        if detected.code and detected.code.split("-")[0] == vote_code.split("-")[0]:
            return vote_code
    if detected and detected.code:
        if preferred_code and detected.code.split("-")[0] == preferred_code.split("-")[0]:
            return preferred_code
        return detected.code
    if preferred_code:
        return preferred_code
    fallback_code = normalize_language_code(fallback)
    return fallback_code


__all__ = ["LanguageGuess", "choose_language", "detect_language", "normalize_language_code"]
