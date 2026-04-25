"""SQLAlchemy async database setup."""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from agentic_rag_os.config import get_os_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


_engine = None
_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_os_settings()
        connect_args = {}
        if settings.is_sqlite:
            connect_args = {"check_same_thread": False}
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            connect_args=connect_args,
        )
    return _engine


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            _get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory


async def init_db() -> None:
    """Create all tables."""
    from agentic_rag_os.models import user, project, reward_config  # noqa: F401
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
