"""
Simple Redis-backed JSON TTL cache with in-memory fallback
"""
from __future__ import annotations

from typing import Any, Optional, Dict, Tuple
import json
import time

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    from redis.asyncio import Redis  # type: ignore
except Exception:  # pragma: no cover
    Redis = None  # type: ignore


class _InMemoryTTL:
    def __init__(self):
        self._store: Dict[str, Tuple[float, str]] = {}

    async def get(self, key: str) -> Optional[str]:
        now = time.time()
        entry = self._store.get(key)
        if not entry:
            return None
        exp, val = entry
        if exp and exp < now:
            try:
                del self._store[key]
            except Exception:
                pass
            return None
        return val

    async def setex(self, key: str, ttl: int, value: str):
        exp = time.time() + max(1, int(ttl))
        self._store[key] = (exp, value)


class RedisTTLCache:
    def __init__(self):
        self._client = None
        self._mem = _InMemoryTTL()
        url = settings.redis_url
        if Redis and url:
            try:
                self._client = Redis.from_url(url, encoding="utf-8", decode_responses=True)
                logger.info("Redis cache initialized", url=url)
            except Exception as e:
                logger.warning("Redis init failed, using in-memory cache", error=str(e))

    async def get_json(self, key: str) -> Optional[Any]:
        try:
            if self._client is not None:
                raw = await self._client.get(key)
            else:
                raw = await self._mem.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.error("Cache get failed", key=key, error=str(e))
            return None

    async def set_json(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        ttl = int(ttl_seconds or settings.quiz_spec_ttl_seconds)
        try:
            raw = json.dumps(value, ensure_ascii=False)
            if self._client is not None:
                await self._client.setex(key, ttl, raw)
            else:
                await self._mem.setex(key, ttl, raw)
        except Exception as e:
            logger.error("Cache set failed", key=key, error=str(e))


_cache_singleton: Optional[RedisTTLCache] = None


def get_cache() -> RedisTTLCache:
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = RedisTTLCache()
    return _cache_singleton

