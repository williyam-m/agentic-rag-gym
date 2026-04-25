"""Configuration for Agentic RAG OS."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class OSSettings(BaseSettings):
    """Settings for the Agentic RAG OS application."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_prefix="RAGAS_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    debug: bool = Field(default=False)
    secret_key: str = Field(
        default="change-me-in-production-use-openssl-rand-hex-32",
        description="JWT signing secret — must be overridden in production.",
    )

    # GitHub OAuth
    github_client_id: str = Field(default="")
    github_client_secret: SecretStr = Field(default=SecretStr(""))
    github_redirect_uri: str = Field(default="http://localhost:8080/api/v1/auth/github/callback")

    # Frontend
    frontend_url: str = Field(default="http://localhost:8080")

    # Database (SQLite by default, PostgreSQL in production)
    database_url: str = Field(default=f"sqlite+aiosqlite:///{BASE_DIR}/agentic_rag_os.db")

    # Storage
    upload_dir: Path = Field(default=BASE_DIR / "agentic_rag_os" / "uploads")
    free_tier_max_mb: int = Field(default=5)      # 5 MB for free tier
    premium_tier_max_mb: int = Field(default=500)  # 500 MB for premium

    # FAISS (reuses rag_master indices dir)
    faiss_base_dir: Path = Field(default=BASE_DIR / "data" / "faiss_indices")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    faiss_dimension: int = Field(default=384)

    # LLM (optional — rewards can be rule-only)
    llm_api_base: str = Field(default="http://localhost:11434/v1")
    llm_api_key: SecretStr = Field(default=SecretStr("none"))
    llm_model: str = Field(default="qwen2.5:7b")

    # JWT
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=10080)  # 7 days

    # CORS
    allowed_origins: list[str] = Field(
        default=["http://localhost:8080", "http://localhost:3000"]
    )

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.database_url


_settings: Optional[OSSettings] = None


def get_os_settings() -> OSSettings:
    global _settings
    if _settings is None:
        _settings = OSSettings()
        _settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return _settings
