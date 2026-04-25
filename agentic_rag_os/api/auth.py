"""GitHub OAuth endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag_os.api.deps import get_current_user
from agentic_rag_os.config import get_os_settings
from agentic_rag_os.database import get_db
from agentic_rag_os.models.user import User
from agentic_rag_os.services.auth_service import (
    build_github_auth_url,
    create_jwt,
    exchange_github_code,
    generate_oauth_state,
    verify_oauth_state,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserProfile"


class UserProfile(BaseModel):
    id: int
    username: str
    display_name: str | None
    email: str | None
    avatar_url: str | None
    tier: str

    model_config = {"from_attributes": True}


TokenResponse.model_rebuild()


@router.get("/github", summary="Initiate GitHub OAuth")
async def github_login():
    """Redirect to GitHub OAuth authorization page."""
    settings = get_os_settings()
    if not settings.github_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub OAuth not configured. Set RAGAS_GITHUB_CLIENT_ID and RAGAS_GITHUB_CLIENT_SECRET.",
        )
    state = generate_oauth_state()
    url = build_github_auth_url(state)
    return RedirectResponse(url=url)


@router.get("/github/callback", summary="GitHub OAuth callback")
async def github_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Handle GitHub OAuth callback, upsert user, return JWT."""
    if not verify_oauth_state(state):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OAuth state")

    try:
        gh_user = await exchange_github_code(code)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"GitHub API error: {exc}",
        ) from exc

    github_id = int(gh_user["id"])
    username = gh_user.get("login", "")
    email = gh_user.get("email")
    display_name = gh_user.get("name") or username
    avatar_url = gh_user.get("avatar_url")

    # Upsert user
    result = await db.execute(select(User).where(User.github_id == github_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(
            github_id=github_id,
            username=username,
            email=email,
            display_name=display_name,
            avatar_url=avatar_url,
        )
        db.add(user)
        await db.flush()
    else:
        user.username = username
        user.email = email
        user.display_name = display_name
        user.avatar_url = avatar_url
        user.last_login = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(user)

    token = create_jwt(user.id, user.username)

    settings = get_os_settings()
    # After OAuth, redirect to frontend with token
    return RedirectResponse(
        url=f"{settings.frontend_url}/auth/callback?token={token}",
        status_code=302,
    )


@router.get("/me", response_model=UserProfile, summary="Get current user profile")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user
