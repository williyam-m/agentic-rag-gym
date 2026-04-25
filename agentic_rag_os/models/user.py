"""User ORM model."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Enum, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentic_rag_os.database import Base


class UserTier(str, PyEnum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    github_id: Mapped[int] = mapped_column(unique=True, index=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bio: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tier: Mapped[UserTier] = mapped_column(Enum(UserTier), default=UserTier.FREE)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    projects: Mapped[List["Project"]] = relationship(  # noqa: F821
        "Project", back_populates="owner", cascade="all, delete-orphan"
    )
    reward_configs: Mapped[List["RewardConfig"]] = relationship(  # noqa: F821
        "RewardConfig", back_populates="owner", cascade="all, delete-orphan"
    )

    @property
    def storage_limit_bytes(self) -> int:
        from agentic_rag_os.config import get_os_settings
        s = get_os_settings()
        if self.tier == UserTier.FREE:
            return s.free_tier_max_mb * 1024 * 1024
        return s.premium_tier_max_mb * 1024 * 1024
