"""Project management endpoints — RAG knowledge bases."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag_os.api.deps import get_current_user
from agentic_rag_os.database import get_db
from agentic_rag_os.models.project import Project, ProjectDocument
from agentic_rag_os.models.user import User, UserTier
from agentic_rag_os.services.rag_service import get_rag_service

router = APIRouter(prefix="/projects", tags=["projects"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: str | None
    doc_count: int
    total_bytes: int
    created_at: str

    model_config = {"from_attributes": True}

    def model_post_init(self, __context):
        if hasattr(self, "created_at") and not isinstance(self.created_at, str):
            object.__setattr__(self, "created_at", self.created_at.isoformat())


class DocumentResponse(BaseModel):
    id: int
    filename: str
    size_bytes: int
    doc_count: int
    created_at: str

    model_config = {"from_attributes": True}


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResult(BaseModel):
    content: str
    score: float
    rank: int
    source: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new RAG project / knowledge base."""
    # Check project limit (free: 3, paid: unlimited)
    result = await db.execute(select(Project).where(Project.owner_id == current_user.id))
    existing = result.scalars().all()
    if current_user.tier == UserTier.FREE and len(existing) >= 3:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Free tier limited to 3 projects. Upgrade to Pro for unlimited projects.",
        )

    from agentic_rag_os.services.rag_service import _project_index_path
    index_path = _project_index_path(current_user.id, 0)  # temp, will update after flush

    project = Project(
        owner_id=current_user.id,
        name=body.name,
        description=body.description,
        index_path=str(index_path),
    )
    db.add(project)
    await db.flush()

    # Update with real project id
    real_index_path = _project_index_path(current_user.id, project.id)
    project.index_path = str(real_index_path)
    await db.commit()
    await db.refresh(project)

    return _project_to_response(project)


@router.get("/", response_model=List[ProjectResponse])
async def list_projects(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Project).where(Project.owner_id == current_user.id).order_by(Project.created_at.desc())
    )
    return [_project_to_response(p) for p in result.scalars().all()]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = await _get_owned_project(project_id, current_user.id, db)
    return _project_to_response(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = await _get_owned_project(project_id, current_user.id, db)
    svc = get_rag_service(current_user.id, project.id)
    await svc.clear()
    await db.delete(project)
    await db.commit()


@router.post("/{project_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    project_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a text document to a project's knowledge base."""
    project = await _get_owned_project(project_id, current_user.id, db)

    content = await file.read()
    size = len(content)

    # Enforce storage limits
    limit = current_user.storage_limit_bytes
    if project.total_bytes + size > limit:
        limit_mb = limit // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Storage limit exceeded ({limit_mb} MB for {current_user.tier} tier). Upgrade to upload more.",
        )

    svc = get_rag_service(current_user.id, project.id)
    try:
        stats = await svc.ingest_file(file.filename or "upload.txt", content)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    doc = ProjectDocument(
        project_id=project.id,
        filename=file.filename or "upload.txt",
        size_bytes=size,
        doc_count=stats["chunks_indexed"],
        storage_path=stats["storage_path"],
    )
    db.add(doc)
    project.total_bytes += size
    project.doc_count += stats["chunks_indexed"]
    await db.commit()
    await db.refresh(doc)

    return {"id": doc.id, "filename": doc.filename, "chunks_indexed": doc.doc_count}


@router.get("/{project_id}/documents", response_model=List[DocumentResponse])
async def list_documents(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = await _get_owned_project(project_id, current_user.id, db)
    result = await db.execute(
        select(ProjectDocument).where(ProjectDocument.project_id == project.id)
    )
    docs = result.scalars().all()
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            size_bytes=d.size_bytes,
            doc_count=d.doc_count,
            created_at=d.created_at.isoformat(),
        )
        for d in docs
    ]


@router.post("/{project_id}/query")
async def query_project(
    project_id: int,
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Query the project's knowledge base using semantic search."""
    project = await _get_owned_project(project_id, current_user.id, db)
    if project.doc_count == 0:
        return {"results": [], "message": "No documents indexed yet."}

    svc = get_rag_service(current_user.id, project.id)
    results = await svc.query(body.query, top_k=body.top_k)
    return {"results": results, "query": body.query, "count": len(results)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_owned_project(project_id: int, owner_id: int, db: AsyncSession) -> Project:
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.owner_id == owner_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


def _project_to_response(p: Project) -> ProjectResponse:
    return ProjectResponse(
        id=p.id,
        name=p.name,
        description=p.description,
        doc_count=p.doc_count,
        total_bytes=p.total_bytes,
        created_at=p.created_at.isoformat(),
    )
