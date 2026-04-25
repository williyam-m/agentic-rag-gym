"""Database models and helpers — SQLite via aiosqlite."""

from __future__ import annotations

import aiosqlite
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_rag_os.config import get_settings

_DB: Optional[aiosqlite.Connection] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    password_hash TEXT,
    github_id TEXT UNIQUE,
    avatar_url TEXT DEFAULT '',
    role TEXT DEFAULT 'user',
    tier TEXT DEFAULT 'free',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    key_hash TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_used TEXT,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS domains (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(user_id, name)
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    domain_id TEXT NOT NULL REFERENCES domains(id),
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reward_configs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    domain_id TEXT REFERENCES domains(id),
    name TEXT NOT NULL,
    algorithm TEXT NOT NULL DEFAULT 'grpo',
    config TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reward_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    reward_config_id TEXT NOT NULL REFERENCES reward_configs(id),
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS rag_queries (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    domain_id TEXT NOT NULL REFERENCES domains(id),
    query TEXT NOT NULL,
    results TEXT DEFAULT '[]',
    reward_score REAL,
    created_at TEXT NOT NULL
);
"""


async def get_db() -> aiosqlite.Connection:
    """Get or create the database connection."""
    global _DB
    if _DB is None:
        settings = get_settings()
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        _DB = await aiosqlite.connect(str(settings.db_path))
        _DB.row_factory = aiosqlite.Row
        await _DB.executescript(SCHEMA)
        await _DB.commit()
    return _DB


async def close_db() -> None:
    global _DB
    if _DB is not None:
        await _DB.close()
        _DB = None


# --- Helper functions ---

def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def fetch_one(query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    db = await get_db()
    cursor = await db.execute(query, params)
    row = await cursor.fetchone()
    if row is None:
        return None
    return dict(row)


async def fetch_all(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    db = await get_db()
    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def execute(query: str, params: tuple = ()) -> None:
    db = await get_db()
    await db.execute(query, params)
    await db.commit()
