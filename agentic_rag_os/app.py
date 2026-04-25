"""
Agentic RAG OS — Main FastAPI Application

Runs independently on port 8080.
Does NOT affect the existing agentic-rag-gym server on port 7860.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agentic_rag_os.config import get_os_settings
from agentic_rag_os.database import init_db

FRONTEND_DIR = Path(__file__).parent / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize database on startup."""
    await init_db()
    yield


def create_app() -> FastAPI:
    settings = get_os_settings()

    app = FastAPI(
        title="Agentic RAG OS",
        description=(
            "RL B2B Rewards-as-a-Service platform — RAG pipeline, dynamic reward generation, "
            "and RL training data for enterprise AI teams."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins + ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routers
    from agentic_rag_os.api.auth import router as auth_router
    from agentic_rag_os.api.projects import router as projects_router
    from agentic_rag_os.api.rewards import router as rewards_router
    from agentic_rag_os.api.users import router as users_router

    API_PREFIX = "/api/v1"
    app.include_router(auth_router, prefix=API_PREFIX)
    app.include_router(users_router, prefix=API_PREFIX)
    app.include_router(projects_router, prefix=API_PREFIX)
    app.include_router(rewards_router, prefix=API_PREFIX)

    # Static files (CSS, JS, images)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Health check
    @app.get("/api/health", tags=["system"])
    async def health():
        return {"status": "ok", "service": "agentic-rag-os", "version": "1.0.0"}

    # SPA routes — serve index.html for all non-API paths
    @app.get("/", response_class=HTMLResponse)
    async def landing(request: Request):
        html_file = TEMPLATES_DIR / "index.html"
        if html_file.exists():
            return HTMLResponse(html_file.read_text())
        return HTMLResponse("<h1>Agentic RAG OS</h1>")

    @app.get("/dashboard", response_class=HTMLResponse)
    @app.get("/dashboard/{rest:path}", response_class=HTMLResponse)
    async def dashboard(request: Request):
        html_file = TEMPLATES_DIR / "dashboard.html"
        if html_file.exists():
            return HTMLResponse(html_file.read_text())
        return HTMLResponse("<h1>Dashboard</h1>")

    @app.get("/auth/callback", response_class=HTMLResponse)
    async def auth_callback(request: Request):
        html_file = TEMPLATES_DIR / "auth_callback.html"
        if html_file.exists():
            return HTMLResponse(html_file.read_text())
        # Fallback: redirect to dashboard
        return HTMLResponse(
            '<script>const p=new URLSearchParams(location.search);'
            'if(p.get("token")){localStorage.setItem("ragas_token",p.get("token"));'
            'location.href="/dashboard";}</script>'
        )

    # Catch-all: serve index.html for SPA navigation
    @app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
    async def catch_all(full_path: str):
        # Don't override API routes
        if full_path.startswith("api/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        html_file = TEMPLATES_DIR / "index.html"
        if html_file.exists():
            return HTMLResponse(html_file.read_text())
        return HTMLResponse("<h1>Page Not Found</h1>", status_code=404)

    return app


app = create_app()
