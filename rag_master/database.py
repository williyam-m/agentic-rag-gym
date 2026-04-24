"""Database adapter for MySQL/SQLAlchemy persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from rag_master.logging_config import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class EpisodeRecord(Base):
    """Persisted episode record."""

    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(String(32), unique=True, nullable=False, index=True)
    task_id = Column(String(64), nullable=False, index=True)
    difficulty = Column(String(16), nullable=False)
    total_steps = Column(Integer, default=0)
    final_score = Column(Float, default=0.0)
    total_reward = Column(Float, default=0.0)
    final_answer = Column(Text, default="")
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)


class StepLog(Base):
    """Persisted step log for trajectory analysis."""

    __tablename__ = "step_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(String(32), nullable=False, index=True)
    step_index = Column(Integer, nullable=False)
    action_type = Column(String(32), nullable=False)
    reasoning_trace = Column(Text, default="")
    intermediate_reward = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DatabaseAdapter:
    """Adapter for database operations using SQLAlchemy."""

    def __init__(self, dsn: str) -> None:
        self._engine = create_engine(dsn, echo=False, pool_pre_ping=True)
        self._session_factory = sessionmaker(bind=self._engine)

    def create_tables(self) -> None:
        """Create database tables."""
        Base.metadata.create_all(self._engine)
        logger.info("database_tables_created")

    def save_episode(self, record: Dict[str, Any]) -> None:
        """Save an episode record."""
        with self._session_factory() as session:
            ep = EpisodeRecord(
                episode_id=record["episode_id"],
                task_id=record["task_id"],
                difficulty=record.get("difficulty", "medium"),
                total_steps=record.get("total_steps", 0),
                final_score=record.get("final_score", 0.0),
                total_reward=record.get("total_reward", 0.0),
                final_answer=record.get("final_answer", ""),
                metadata_json=record.get("metadata", {}),
                completed_at=datetime.now(timezone.utc),
            )
            session.add(ep)
            session.commit()

    def save_step(self, step: Dict[str, Any]) -> None:
        """Save a step log."""
        with self._session_factory() as session:
            log = StepLog(
                episode_id=step["episode_id"],
                step_index=step["step_index"],
                action_type=step["action_type"],
                reasoning_trace=step.get("reasoning_trace", ""),
                intermediate_reward=step.get("intermediate_reward", 0.0),
            )
            session.add(log)
            session.commit()

    def get_episodes(
        self, task_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve episode records."""
        with self._session_factory() as session:
            query = session.query(EpisodeRecord)
            if task_id:
                query = query.filter(EpisodeRecord.task_id == task_id)
            query = query.order_by(EpisodeRecord.created_at.desc()).limit(limit)
            results = []
            for ep in query.all():
                results.append({
                    "episode_id": ep.episode_id,
                    "task_id": ep.task_id,
                    "difficulty": ep.difficulty,
                    "total_steps": ep.total_steps,
                    "final_score": ep.final_score,
                    "total_reward": ep.total_reward,
                    "created_at": str(ep.created_at),
                })
            return results

    def get_steps(self, episode_id: str) -> List[Dict[str, Any]]:
        """Retrieve step logs for an episode."""
        with self._session_factory() as session:
            query = (
                session.query(StepLog)
                .filter(StepLog.episode_id == episode_id)
                .order_by(StepLog.step_index)
            )
            return [
                {
                    "step_index": s.step_index,
                    "action_type": s.action_type,
                    "reasoning_trace": s.reasoning_trace,
                    "intermediate_reward": s.intermediate_reward,
                }
                for s in query.all()
            ]
