"""User management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from agentic_rag_os.api.deps import get_current_user
from agentic_rag_os.models.user import User

router = APIRouter(prefix="/users", tags=["users"])


class UserProfile(BaseModel):
    id: int
    username: str
    display_name: str | None
    email: str | None
    avatar_url: str | None
    tier: str
    is_active: bool
    created_at: str

    model_config = {"from_attributes": True}


@router.get("/me", response_model=UserProfile, summary="Get own profile")
async def get_my_profile(current_user: User = Depends(get_current_user)):
    return UserProfile(
        id=current_user.id,
        username=current_user.username,
        display_name=current_user.display_name,
        email=current_user.email,
        avatar_url=current_user.avatar_url,
        tier=current_user.tier.value if hasattr(current_user.tier, "value") else current_user.tier,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat(),
    )


@router.get("/me/storage", summary="Get storage usage")
async def get_storage_usage(current_user: User = Depends(get_current_user)):
    from agentic_rag_os.config import get_os_settings
    settings = get_os_settings()
    limit_bytes = current_user.storage_limit_bytes
    from agentic_rag_os.models.user import UserTier
    tier = current_user.tier
    return {
        "tier": tier.value if hasattr(tier, "value") else tier,
        "limit_bytes": limit_bytes,
        "limit_mb": limit_bytes // (1024 * 1024),
        "free_limit_mb": settings.free_tier_max_mb,
        "premium_limit_mb": settings.premium_tier_max_mb,
    }
