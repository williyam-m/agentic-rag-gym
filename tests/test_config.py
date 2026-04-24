"""Tests for configuration management."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from rag_master.config import AppSettings, EvaluationMode, LogLevel, get_settings


class TestAppSettings:
    """Test suite for AppSettings."""

    def test_default_values(self) -> None:
        """Test that defaults are sensible."""
        settings = AppSettings(
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.server_port == 7860
        assert settings.faiss_dimension == 384
        assert settings.env_max_steps == 20
        assert settings.log_level == LogLevel.INFO

    def test_mysql_dsn_construction(self) -> None:
        """Test MySQL DSN is properly constructed."""
        settings = AppSettings(
            mysql_user="testuser",
            mysql_password="testpass",  # type: ignore[arg-type]
            mysql_host="db.example.com",
            mysql_port=3307,
            mysql_database="testdb",
            _env_file=None,  # type: ignore[call-arg]
        )
        dsn = settings.mysql_dsn
        assert "testuser" in dsn
        assert "testpass" in dsn
        assert "db.example.com" in dsn
        assert "3307" in dsn
        assert "testdb" in dsn

    def test_llm_api_key_priority(self) -> None:
        """Test API key selection priority."""
        settings = AppSettings(
            hf_token="hf_test",  # type: ignore[arg-type]
            groq_api_key="gsk_test",  # type: ignore[arg-type]
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.llm_api_key == "hf_test"

    def test_llm_api_key_fallback_groq(self) -> None:
        """Test GROQ fallback when HF token is empty."""
        settings = AppSettings(
            hf_token="",  # type: ignore[arg-type]
            groq_api_key="gsk_test",  # type: ignore[arg-type]
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.llm_api_key == "gsk_test"

    def test_llm_api_key_fallback_ollama(self) -> None:
        """Test Ollama fallback when both keys are empty."""
        settings = AppSettings(
            hf_token="",  # type: ignore[arg-type]
            groq_api_key="",  # type: ignore[arg-type]
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.llm_api_key == "ollama"

    def test_port_validation(self) -> None:
        """Test port validation bounds."""
        with pytest.raises(Exception):
            AppSettings(server_port=0, _env_file=None)  # type: ignore[call-arg]
        with pytest.raises(Exception):
            AppSettings(server_port=70000, _env_file=None)  # type: ignore[call-arg]

    def test_evaluation_mode_enum(self) -> None:
        """Test EvaluationMode enum values."""
        assert EvaluationMode.RULE_BASED == "rule_based"
        assert EvaluationMode.LLM_JUDGE == "llm_judge"
        assert EvaluationMode.HYBRID == "hybrid"

    def test_get_settings_factory(self) -> None:
        """Test settings factory function."""
        settings = get_settings()
        assert isinstance(settings, AppSettings)
