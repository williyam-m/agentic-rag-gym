"""Authentication service — JWT + password hashing."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import jwt

from agentic_rag_os.config import get_settings
from agentic_rag_os.models import fetch_one, execute, new_id, now_iso


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310_000)
    return f"{salt}${h.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    salt, h = stored.split("$", 1)
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310_000)
    return hmac.compare_digest(candidate.hex(), h)


def _create_token(user_id: str, username: str, role: str = "user") -> str:
    settings = get_settings()
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> Dict[str, Any]:
    settings = get_settings()
    return jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])


async def register_user(username: str, email: str, password: str) -> Dict[str, Any]:
    existing = await fetch_one("SELECT id FROM users WHERE username=? OR email=?", (username, email))
    if existing:
        raise ValueError("Username or email already exists")

    uid = new_id()
    ts = now_iso()
    pw_hash = _hash_password(password)
    await execute(
        "INSERT INTO users (id, username, email, password_hash, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (uid, username, email, pw_hash, ts, ts),
    )
    user = await fetch_one("SELECT * FROM users WHERE id=?", (uid,))
    token = _create_token(uid, username)
    return {"access_token": token, "token_type": "bearer", "user": _user_out(user)}


async def login_user(username: str, password: str) -> Dict[str, Any]:
    user = await fetch_one("SELECT * FROM users WHERE username=?", (username,))
    if not user or not user.get("password_hash"):
        raise ValueError("Invalid credentials")
    if not _verify_password(password, user["password_hash"]):
        raise ValueError("Invalid credentials")
    token = _create_token(user["id"], user["username"], user.get("role", "user"))
    return {"access_token": token, "token_type": "bearer", "user": _user_out(user)}


async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    user = await fetch_one("SELECT * FROM users WHERE id=?", (user_id,))
    if user:
        return _user_out(user)
    return None


def _user_out(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user.get("email", ""),
        "avatar_url": user.get("avatar_url", ""),
        "role": user.get("role", "user"),
        "tier": user.get("tier", "free"),
        "created_at": user["created_at"],
    }


# --- API Key management ---

def _hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def create_api_key(user_id: str, name: str) -> Dict[str, Any]:
    key = f"ragos_{secrets.token_hex(24)}"
    kid = new_id()
    ts = now_iso()
    await execute(
        "INSERT INTO api_keys (id, user_id, key_hash, name, created_at) VALUES (?,?,?,?,?)",
        (kid, user_id, _hash_api_key(key), name, ts),
    )
    return {"id": kid, "name": name, "key": key, "created_at": ts}


async def validate_api_key(key: str) -> Optional[Dict[str, Any]]:
    h = _hash_api_key(key)
    row = await fetch_one(
        "SELECT ak.*, u.username, u.role FROM api_keys ak JOIN users u ON ak.user_id=u.id WHERE ak.key_hash=? AND ak.is_active=1",
        (h,),
    )
    if row:
        await execute("UPDATE api_keys SET last_used=? WHERE id=?", (now_iso(), row["id"]))
        return {"user_id": row["user_id"], "username": row["username"], "role": row["role"]}
    return None


async def list_api_keys(user_id: str):
    from agentic_rag_os.models import fetch_all
    rows = await fetch_all(
        "SELECT id, name, created_at, last_used, is_active FROM api_keys WHERE user_id=?",
        (user_id,),
    )
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "key_prefix": "ragos_****",
            "created_at": r["created_at"],
            "last_used": r.get("last_used"),
            "is_active": bool(r["is_active"]),
        }
        for r in rows
    ]


async def revoke_api_key(user_id: str, key_id: str) -> bool:
    row = await fetch_one("SELECT id FROM api_keys WHERE id=? AND user_id=?", (key_id, user_id))
    if not row:
        return False
    await execute("UPDATE api_keys SET is_active=0 WHERE id=?", (key_id,))
    return True
