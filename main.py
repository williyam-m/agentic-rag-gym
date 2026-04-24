"""Main entry point for the Agentic RAG Gym server."""

from __future__ import annotations

import os
import sys

import gradio as gr
import uvicorn

from rag_master.config import get_settings
from rag_master.logging_config import setup_logging
from server.app import app as fastapi_app
from server.ui import build_ui


def main() -> None:
    """Start the Agentic RAG Gym server with Gradio UI mounted on FastAPI."""
    settings = get_settings()
    setup_logging(settings.log_level.value)

    # Build and mount Gradio UI at /ui (not / which would catch all routes)
    ui = build_ui()
    gr.mount_gradio_app(fastapi_app, ui, path="/ui")

    # Redirect root to /ui
    from fastapi.responses import RedirectResponse

    @fastapi_app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/ui")

    uvicorn.run(
        fastapi_app,
        host=settings.server_host,
        port=settings.server_port,
        log_level=settings.log_level.value.lower(),
    )


if __name__ == "__main__":
    main()
