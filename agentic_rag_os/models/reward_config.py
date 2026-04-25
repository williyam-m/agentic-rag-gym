"""Reward configuration ORM model — Rewards-as-a-Service core entity."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Any, Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentic_rag_os.database import Base


class RewardAlgorithm(str, PyEnum):
    GRPO = "grpo"
    PPO = "ppo"
    DPO = "dpo"
    CUSTOM = "custom"


class RewardStatus(str, PyEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class RewardConfig(Base):
    """A user-defined reward function configuration."""

    __tablename__ = "reward_configs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(256))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    algorithm: Mapped[RewardAlgorithm] = mapped_column(
        Enum(RewardAlgorithm), default=RewardAlgorithm.GRPO
    )
    # JSON-encoded list of RewardRule objects
    rules: Mapped[Any] = mapped_column(JSON, default=list)
    # JSON-encoded weight map {rule_name: weight}
    weights: Mapped[Any] = mapped_column(JSON, default=dict)
    status: Mapped[RewardStatus] = mapped_column(
        Enum(RewardStatus), default=RewardStatus.DRAFT
    )
    compute_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    owner: Mapped["User"] = relationship("User", back_populates="reward_configs")  # noqa: F821
    computations: Mapped[list["RewardComputation"]] = relationship(
        "RewardComputation", back_populates="config", cascade="all, delete-orphan"
    )


class RewardComputation(Base):
    """A single batch reward computation run."""

    __tablename__ = "reward_computations"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    config_id: Mapped[int] = mapped_column(ForeignKey("reward_configs.id"), index=True)
    input_count: Mapped[int] = mapped_column(Integer, default=0)
    # JSON-encoded list of {prompt, response, reward, breakdown}
    results: Mapped[Any] = mapped_column(JSON, default=list)
    avg_reward: Mapped[Optional[float]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    config: Mapped["RewardConfig"] = relationship(
        "RewardConfig", back_populates="computations"
    )
