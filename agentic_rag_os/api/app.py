"""Main FastAPI application for Agentic RAG OS."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agentic_rag_os.config import get_settings
from agentic_rag_os.models import close_db, get_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Startup/shutdown lifecycle."""
    await get_db()
    yield
    await close_db()


app = FastAPI(
    title="Agentic RAG OS",
    description="The Operating System for Agentic RAG — RL B2B Rewards-as-a-Service",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register API routers ---
from agentic_rag_os.api.auth_routes import router as auth_router
from agentic_rag_os.api.rag_routes import router as rag_router
from agentic_rag_os.api.reward_routes import router as reward_router
from agentic_rag_os.api.user_routes import router as user_router

app.include_router(auth_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")
app.include_router(reward_router, prefix="/api/v1")
app.include_router(user_router, prefix="/api/v1")


# --- Health check ---
@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "service": "agentic-rag-os", "version": "1.0.0"}


# --- Serve static files ---
_FRONTEND_DIR = Path(__file__).parent / "frontend"
_STATIC_DIR = _FRONTEND_DIR / "static"

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# --- Serve SPA ---
@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    """Serve the single-page application for all non-API routes."""
    # Don't intercept API paths
    if full_path.startswith("api/"):
        return JSONResponse({"detail": "Not found"}, status_code=404)

    index = _FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return HTMLResponse("<h1>Agentic RAG OS</h1><p>Frontend not built yet.</p>")


# --- Global error handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )
