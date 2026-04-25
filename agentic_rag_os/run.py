"""Entry point for Agentic RAG OS application."""

from __future__ import annotations

import uvicorn

from agentic_rag_os.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "agentic_rag_os.api.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
