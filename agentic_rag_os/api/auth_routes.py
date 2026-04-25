"""Authentication routes — register, login, GitHub OAuth."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from agentic_rag_os.models.schemas import LoginRequest, RegisterRequest, TokenResponse
from agentic_rag_os.services import github_oauth_callback, login_user, register_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    try:
        result = await register_user(req.username, req.email, req.password)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    try:
        result = await login_user(req.username, req.password)
        return result
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.get("/github/callback")
async def github_callback(code: str = Query(...)):
    """GitHub OAuth callback — exchange code for JWT."""
    try:
        result = await github_oauth_callback(code)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
