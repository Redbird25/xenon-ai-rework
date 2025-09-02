from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.config import DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=True, future=True)
async_session = async_sessionmaker(engine, expire_on_commit=False)
