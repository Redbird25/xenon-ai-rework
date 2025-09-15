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

    async def exists(self, key: str) -> bool:
        return (await self.get(key)) is not None

    async def delete(self, key: str):
        try:
            if key in self._store:
                del self._store[key]
        except Exception:
            pass

    async def setnx(self, key: str, ttl: int, value: str) -> bool:
        now = time.time()
        entry = self._store.get(key)
        if entry:
            exp, _ = entry
            if exp and exp > now:
                return False
        await self.setex(key, ttl, value)
        return True


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
        else:
            if not url:
                logger.warning("REDIS_URL not configured. Using in-memory TTL cache (non-persistent)")
            elif not Redis:
                logger.warning("redis-py is not available. Using in-memory TTL cache (non-persistent)")

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

    async def exists(self, key: str) -> bool:
        try:
            if self._client is not None:
                val = await self._client.exists(key)
                return bool(val)
            return await self._mem.exists(key)
        except Exception as e:
            logger.error("Cache exists failed", key=key, error=str(e))
            return False

    async def delete(self, key: str):
        try:
            if self._client is not None:
                await self._client.delete(key)
            else:
                await self._mem.delete(key)
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))

    async def set_if_not_exists(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        ttl = int(ttl_seconds or settings.quiz_spec_ttl_seconds)
        try:
            raw = json.dumps(value, ensure_ascii=False)
            if self._client is not None:
                # SET key value NX EX ttl
                res = await self._client.set(key, raw, ex=ttl, nx=True)
                return bool(res)
            else:
                return await self._mem.setnx(key, ttl, raw)
        except Exception as e:
            logger.error("Cache set_if_not_exists failed", key=key, error=str(e))
            return False


_cache_singleton: Optional[RedisTTLCache] = None


def get_cache() -> RedisTTLCache:
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = RedisTTLCache()
    return _cache_singleton

