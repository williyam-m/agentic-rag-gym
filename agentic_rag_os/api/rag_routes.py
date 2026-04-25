"""RAG pipeline routes — domains, documents, queries."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from agentic_rag_os.api.deps import get_current_user
from agentic_rag_os.models.schemas import (
    DomainCreate,
    DomainOut,
    DocumentOut,
    RAGQueryRequest,
    RAGQueryResponse,
)
from agentic_rag_os.services.rag_service import (
    create_domain,
    delete_document,
    delete_domain,
    list_documents,
    list_domains,
    rag_query,
    upload_document,
)

router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])


# --- Domains ---

@router.post("/domains", response_model=DomainOut)
async def create_domain_route(body: DomainCreate, user: Dict = Depends(get_current_user)):
    return await create_domain(user["user_id"], body.name, body.description)


@router.get("/domains", response_model=List[DomainOut])
async def list_domains_route(user: Dict = Depends(get_current_user)):
    return await list_domains(user["user_id"])


@router.delete("/domains/{domain_id}")
async def delete_domain_route(domain_id: str, user: Dict = Depends(get_current_user)):
    ok = await delete_domain(user["user_id"], domain_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Domain not found")
    return {"ok": True}


# --- Documents ---

@router.get("/domains/{domain_id}/documents", response_model=List[DocumentOut])
async def list_docs_route(domain_id: str, user: Dict = Depends(get_current_user)):
    return await list_documents(user["user_id"], domain_id)


@router.post("/domains/{domain_id}/documents", response_model=DocumentOut)
async def upload_doc_route(domain_id: str, file: UploadFile = File(...), user: Dict = Depends(get_current_user)):
    """Upload a text file to a domain. Free tier: max 2MB total."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")
    content = (await file.read()).decode("utf-8", errors="replace")
    if not content.strip():
        raise HTTPException(status_code=400, detail="File is empty")
    try:
        return await upload_document(user["user_id"], domain_id, file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/domains/{domain_id}/documents/{doc_id}")
async def delete_doc_route(domain_id: str, doc_id: str, user: Dict = Depends(get_current_user)):
    ok = await delete_document(user["user_id"], domain_id, doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"ok": True}


# --- RAG Query ---

@router.post("/domains/{domain_id}/query", response_model=RAGQueryResponse)
async def query_route(domain_id: str, body: RAGQueryRequest, user: Dict = Depends(get_current_user)):
    try:
        return await rag_query(user["user_id"], domain_id, body.query, body.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
