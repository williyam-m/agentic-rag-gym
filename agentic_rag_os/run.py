"""Entry point for Agentic RAG OS server."""

from __future__ import annotations

import uvicorn

from agentic_rag_os.config import get_os_settings


def main() -> None:
    settings = get_os_settings()
    uvicorn.run(
        "agentic_rag_os.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
