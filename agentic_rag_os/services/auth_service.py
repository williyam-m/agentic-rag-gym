"""GitHub OAuth and JWT authentication service."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx
from jose import JWTError, jwt

from agentic_rag_os.config import get_os_settings

_STATE_STORE: Dict[str, float] = {}  # state → expiry timestamp


def generate_oauth_state() -> str:
    """Generate a cryptographically secure OAuth state parameter."""
    import time
    settings = get_os_settings()
    state = secrets.token_urlsafe(32)
    _STATE_STORE[state] = time.time() + 600  # 10-minute expiry
    return state


def verify_oauth_state(state: str) -> bool:
    """Verify and consume OAuth state. Returns True if valid."""
    import time
    expiry = _STATE_STORE.pop(state, None)
    if expiry is None:
        return False
    return time.time() < expiry


def build_github_auth_url(state: str) -> str:
    """Build GitHub OAuth authorization URL."""
    settings = get_os_settings()
    params = (
        f"client_id={settings.github_client_id}"
        f"&redirect_uri={settings.github_redirect_uri}"
        f"&scope=read:user+user:email"
        f"&state={state}"
    )
    return f"https://github.com/login/oauth/authorize?{params}"


async def exchange_github_code(code: str) -> Dict[str, Any]:
    """Exchange OAuth code for access token and return GitHub user info."""
    settings = get_os_settings()
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Exchange code for token
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret.get_secret_value(),
                "code": code,
                "redirect_uri": settings.github_redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
        token_resp.raise_for_status()
        token_data = token_resp.json()
        access_token = token_data.get("access_token", "")
        if not access_token:
            raise ValueError(f"No access token in response: {token_data}")

        # Fetch user info
        user_resp = await client.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
            },
        )
        user_resp.raise_for_status()
        user_data = user_resp.json()

        # Fetch email if not in profile
        if not user_data.get("email"):
            email_resp = await client.get(
                "https://api.github.com/user/emails",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github+json",
                },
            )
            if email_resp.status_code == 200:
                emails = email_resp.json()
                primary = next(
                    (e["email"] for e in emails if e.get("primary") and e.get("verified")),
                    None,
                )
                user_data["email"] = primary

        return user_data


def create_jwt(user_id: int, username: str) -> str:
    """Create a signed JWT for the given user."""
    settings = get_os_settings()
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_jwt(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify a JWT. Returns payload or None."""
    settings = get_os_settings()
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError:
        return None
