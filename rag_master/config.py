"""Core configuration management using Pydantic Settings."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvaluationMode(str, Enum):
    """Evaluation strategy for the RAG pipeline."""

    RULE_BASED = "rule_based"
    LLM_JUDGE = "llm_judge"
    HYBRID = "hybrid"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class AppSettings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    api_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for the OpenAI-compatible LLM API.",
    )
    model_name: str = Field(
        default="qwen2.5:7b",
        description="Model identifier for inference.",
    )
    hf_token: SecretStr = Field(
        default=SecretStr(""),
        description="Hugging Face API token.",
    )
    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="GROQ API key.",
    )

    # --- Database ---
    mysql_host: str = Field(default="localhost")
    mysql_port: int = Field(default=3306, ge=1, le=65535)
    mysql_user: str = Field(default="raguser")
    mysql_password: SecretStr = Field(default=SecretStr("changeme_secure_password"))
    mysql_database: str = Field(default="agentic_rag_gym")

    # --- FAISS ---
    faiss_index_dir: Path = Field(default=Path("./data/faiss_indices"))
    faiss_dimension: int = Field(default=384, ge=1)

    # --- Server ---
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=7860, ge=1, le=65535)
    log_level: LogLevel = Field(default=LogLevel.INFO)

    # --- Embedding ---
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # --- Environment ---
    env_max_steps: int = Field(default=20, ge=1)
    env_timeout_seconds: int = Field(default=300, ge=1)

    @field_validator("faiss_index_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Convert string to Path and create directory if needed."""
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def mysql_dsn(self) -> str:
        """Build MySQL connection string."""
        pwd = self.mysql_password.get_secret_value()
        return (
            f"mysql+pymysql://{self.mysql_user}:{pwd}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    @property
    def llm_api_key(self) -> str:
        """Return the best available API key."""
        hf = self.hf_token.get_secret_value()
        if hf:
            return hf
        groq = self.groq_api_key.get_secret_value()
        if groq:
            return groq
        return "ollama"


def get_settings() -> AppSettings:
    """Factory for application settings (cacheable)."""
    return AppSettings()
