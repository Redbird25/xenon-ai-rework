"""
Database configuration with connection pooling and async support
"""
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import text
from app.config import settings

# Create async engine with appropriate pooling
engine_kwargs = {
    "echo": settings.database_echo,
    "future": True,
    "pool_pre_ping": True,
}

if settings.is_production:
    # Use QueuePool in production with explicit sizing
    engine_kwargs.update({
        "poolclass": QueuePool,
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_pool_size * 2,
    })
else:
    # Use NullPool in dev to avoid invalid pool args and stale conns
    engine_kwargs.update({
        "poolclass": NullPool,
    })

engine = create_async_engine(
    settings.database_url,
    **engine_kwargs
)

# Create async session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)
