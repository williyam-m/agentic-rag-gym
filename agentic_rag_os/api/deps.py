"""Authentication dependency for FastAPI routes."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agentic_rag_os.services import decode_token, validate_api_key

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Dict[str, Any]:
    """Extract and validate the current user from JWT or API key."""

    # Try Bearer token (JWT)
    if credentials and credentials.credentials:
        try:
            payload = decode_token(credentials.credentials)
            return {"user_id": payload["sub"], "username": payload["username"], "role": payload.get("role", "user")}
        except Exception:
            pass

    # Try API key in X-API-Key header
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        user_info = await validate_api_key(api_key)
        if user_info:
            return user_info

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Provide a Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[Dict[str, Any]]:
    """Optionally get the current user — returns None if not authenticated."""
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None
