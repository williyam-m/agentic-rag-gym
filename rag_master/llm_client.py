"""OpenAI-compatible LLM client implementation."""

from __future__ import annotations

from typing import Any, Dict, List

from openai import AsyncOpenAI

from rag_master.adapters import BaseLLMClient
from rag_master.logging_config import get_logger

logger = get_logger(__name__)


class OpenAICompatibleClient(BaseLLMClient):
    """LLM client that works with any OpenAI-compatible API (Ollama, GROQ, HF, etc.)."""

    def __init__(self, base_url: str, api_key: str, model_name: str, timeout: float = 30.0) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._model = model_name

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from the LLM."""
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = completion.choices[0].message.content or ""
            return content.strip()
        except Exception as exc:
            logger.error("llm_generation_failed", error=str(exc))
            raise

    async def generate_with_metadata(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Generate a response with token usage metadata."""
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = completion.choices[0].message.content or ""
            usage = completion.usage
            return {
                "content": content.strip(),
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "model": completion.model,
                "finish_reason": completion.choices[0].finish_reason,
            }
        except Exception as exc:
            logger.error("llm_generation_failed", error=str(exc))
            raise
