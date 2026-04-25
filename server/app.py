"""
FastAPI server implementing the OpenEnv specification for Agentic RAG Gym.

Endpoints:
- POST /reset       → Reset environment, start new episode
- POST /step        → Take an action in the environment
- GET  /state       → Get current environment state
- GET  /tasks       → List available tasks
- POST /grade       → Grade current episode
- GET  /health      → Health check
"""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_master.agents import CriticAgent, PlannerAgent, ReasonerAgent, RetrieverAgent, VerifierAgent
from rag_master.config import AppSettings, get_settings
from rag_master.llm_client import OpenAICompatibleClient
from rag_master.logging_config import get_logger, setup_logging
from rag_master.models import AgentRole
from rag_master.orchestrator import Orchestrator
from rag_master.retriever import FAISSRetriever
from rag_master.rewards import CompositeRewardFunction

from domains.aerospace.config import AerospaceDomainConfig
from domains.legal_research.config import LegalResearchDomainConfig
from server.models import (
    Action,
    GradeResult,
    HealthStatus,
    Observation,
    ResetResult,
    StateResult,
    StepResult,
    TaskInfo,
)

logger = get_logger(__name__)

# --- Domain registry ---
DOMAIN_REGISTRY = {
    "aerospace": AerospaceDomainConfig,
    "legal_research": LegalResearchDomainConfig,
}

# --- Global state ---
_orchestrators: Dict[str, Orchestrator] = {}
_active_domain: str = "aerospace"
_settings: Optional[AppSettings] = None
_lock = asyncio.Lock()

# Regex pattern for sanitization: strip HTML tags and control chars
_SANITIZE_RE = re.compile(r'<[^>]+>|[\x00-\x08\x0b\x0c\x0e-\x1f]')


class ResetRequest(BaseModel):
    """Request body for /reset endpoint."""
    task_id: Optional[str] = Field(default=None, description="Task ID to load.")


class GradeRequest(BaseModel):
    """Request body for /grade endpoint."""
    task_id: Optional[str] = Field(default=None, description="Task ID to grade.")


def _sanitize(text: str) -> str:
    """Sanitize user input: strip HTML tags and control characters."""
    return _SANITIZE_RE.sub('', text).strip()


async def _initialize_orchestrator(settings: AppSettings, domain_name: str = "aerospace") -> Orchestrator:
    """Initialize the orchestrator with all components for a given domain."""
    domain_cls = DOMAIN_REGISTRY.get(domain_name)
    if domain_cls is None:
        raise ValueError(f"Unknown domain: {domain_name}. Available: {list(DOMAIN_REGISTRY.keys())}")
    domain = domain_cls()

    # Each domain gets its own subdirectory so they never share embeddings.
    domain_index_dir = settings.faiss_index_dir / domain_name
    retriever = FAISSRetriever(
        index_dir=domain_index_dir,
        embedding_model=settings.embedding_model,
        dimension=settings.faiss_dimension,
    )

    llm = OpenAICompatibleClient(
        base_url=settings.api_base_url,
        api_key=settings.llm_api_key,
        model_name=settings.model_name,
    )

    reward_fn = CompositeRewardFunction(llm_client=llm)

    agents = {
        AgentRole.RETRIEVER: RetrieverAgent(retriever=retriever, llm=llm),
        AgentRole.REASONER: ReasonerAgent(llm=llm, system_prompt=domain.get_system_prompt()),
        AgentRole.CRITIC: CriticAgent(llm=llm),
        AgentRole.PLANNER: PlannerAgent(llm=llm),
        AgentRole.VERIFIER: VerifierAgent(llm=llm),
    }

    orchestrator = Orchestrator(
        domain_config=domain,
        retriever=retriever,
        llm_client=llm,
        reward_function=reward_fn,
        agents=agents,
    )

    docs = domain.get_documents()
    count = await retriever.index_documents(docs)
    logger.info("knowledge_base_indexed", document_count=count)

    return orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    global _orchestrators, _active_domain, _settings
    _settings = get_settings()
    setup_logging(_settings.log_level.value)
    logger.info("server_starting", host=_settings.server_host, port=_settings.server_port)

    # Initialize default domain (aerospace)
    _active_domain = "aerospace"
    _orchestrators["aerospace"] = await _initialize_orchestrator(_settings, "aerospace")
    logger.info("orchestrator_initialized", domain="aerospace")
    yield
    logger.info("server_shutting_down")


app = FastAPI(
    title="Agentic RAG Gym",
    description="A reinforcement learning environment for training AI agents on Retrieval-Augmented Generation tasks across multiple domains.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Prometheus metrics ---
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
except ImportError:
    logger.warning("prometheus_not_available", msg="Install prometheus_fastapi_instrumentator for /metrics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthStatus)
async def health() -> HealthStatus:
    """Health check endpoint."""
    return HealthStatus()


@app.get("/domains")
async def list_domains() -> Dict[str, Any]:
    """List all available domains and the currently active one."""
    return {
        "domains": list(DOMAIN_REGISTRY.keys()),
        "active": _active_domain,
    }


class SwitchDomainRequest(BaseModel):
    """Request body for /domain/switch endpoint."""
    domain: str = Field(description="Domain name to switch to.")


@app.post("/domain/switch")
async def switch_domain(request: SwitchDomainRequest) -> Dict[str, Any]:
    """Switch the active domain."""
    global _active_domain
    domain_name = _sanitize(request.domain)
    if domain_name not in DOMAIN_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain: {domain_name}. Available: {list(DOMAIN_REGISTRY.keys())}",
        )

    async with _lock:
        if domain_name not in _orchestrators:
            if _settings is None:
                raise HTTPException(status_code=503, detail="Settings not initialized")
            _orchestrators[domain_name] = await _initialize_orchestrator(_settings, domain_name)
            logger.info("orchestrator_initialized", domain=domain_name)
        _active_domain = domain_name

    return {"active": _active_domain, "message": f"Switched to {domain_name} domain"}


def _get_orchestrator() -> Orchestrator:
    """Get the currently active orchestrator."""
    orch = _orchestrators.get(_active_domain)
    if orch is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orch


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    """Reset the environment for a new episode."""
    orchestrator = _get_orchestrator()

    task_id = _sanitize(request.task_id) if request.task_id else None
    try:
        async with _lock:
            observation = await orchestrator.reset(task_id=task_id)
        return {"observation": observation, "done": False, "info": {"message": "Episode reset"}}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
async def step(action: Action) -> Dict[str, Any]:
    """Take an action in the environment."""
    orchestrator = _get_orchestrator()

    action_dict: Dict[str, Any] = {"type": action.type.value}
    if action.query:
        action_dict["query"] = _sanitize(action.query)
    if action.answer:
        action_dict["answer"] = _sanitize(action.answer)
    # Merge extra parameters but block overriding type/query/answer
    for k, v in action.parameters.items():
        if k not in ("type", "query", "answer"):
            action_dict[k] = v

    try:
        async with _lock:
            result = await orchestrator.step(action_dict)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
async def state() -> Dict[str, Any]:
    """Get the current environment state."""
    orchestrator = _get_orchestrator()
    return orchestrator.state()


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """List all available tasks."""
    orchestrator = _get_orchestrator()
    tasks = [
        TaskInfo(
            task_id=t.task_id,
            name=t.name,
            description=t.description,
            difficulty=t.difficulty.value,
            max_steps=t.max_steps,
        ).model_dump()
        for t in orchestrator.tasks
    ]
    return {"tasks": tasks}


@app.post("/grade")
async def grade(request: GradeRequest = GradeRequest()) -> Dict[str, Any]:
    """Grade the current episode."""
    orchestrator = _get_orchestrator()

    try:
        async with _lock:
            score = await orchestrator.grade(task_id=request.task_id)
        state_data = orchestrator.state()
        return GradeResult(
            task_id=state_data.get("task_id", ""),
            score=score,
            episode_id=state_data.get("episode_id", ""),
        ).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
