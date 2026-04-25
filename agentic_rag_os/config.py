"""Configuration for Agentic RAG OS."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

_ROOT = Path(__file__).resolve().parent.parent


class RagOSSettings(BaseSettings):
    """Central settings for the Agentic RAG OS application."""

    # --- Application ---
    app_name: str = "Agentic RAG OS"
    secret_key: str = Field(default_factory=lambda: os.getenv("RAGOS_SECRET_KEY", secrets.token_hex(32)))
    debug: bool = False
    base_url: str = os.getenv("RAGOS_BASE_URL", "http://localhost:8000")

    # --- Auth ---
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 1440  # 24h
    # --- Database (SQLite for simplicity, file-based) ---
    db_path: Path = _ROOT / "agentic_rag_os" / "data" / "ragos.db"

    # --- Upload limits ---
    max_upload_mb: float = 2.0  # Free tier
    premium_upload_mb: float = 100.0

    # --- RAG Master bridge ---
    gym_base_url: str = os.getenv("GYM_BASE_URL", "http://localhost:7860")

    # --- Embedding ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_dimension: int = 384

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "RAGOS_", "env_file": ".env", "extra": "ignore"}


_settings: Optional[RagOSSettings] = None


def get_settings() -> RagOSSettings:
    global _settings
    if _settings is None:
        _settings = RagOSSettings()
    return _settings
