"""User management routes — profile, API keys, dashboard stats."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from agentic_rag_os.api.deps import get_current_user
from agentic_rag_os.models.schemas import APIKeyCreate, APIKeyCreated, APIKeyOut, DashboardStats, UserOut
from agentic_rag_os.services import create_api_key, get_user_by_id, list_api_keys, revoke_api_key
from agentic_rag_os.models import fetch_all, fetch_one
from agentic_rag_os.config import get_settings

router = APIRouter(prefix="/user", tags=["User Management"])


@router.get("/me", response_model=UserOut)
async def get_profile(user: Dict = Depends(get_current_user)):
    u = await get_user_by_id(user["user_id"])
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return u


@router.get("/dashboard", response_model=DashboardStats)
async def dashboard_stats(user: Dict = Depends(get_current_user)):
    uid = user["user_id"]
    settings = get_settings()

    domains = await fetch_all("SELECT id FROM domains WHERE user_id=?", (uid,))
    domain_ids = [d["id"] for d in domains]

    doc_count = 0
    total_size = 0
    if domain_ids:
        placeholders = ",".join("?" for _ in domain_ids)
        docs = await fetch_all(f"SELECT size_bytes FROM documents WHERE domain_id IN ({placeholders})", tuple(domain_ids))
        doc_count = len(docs)
        total_size = sum(d["size_bytes"] for d in docs)

    queries = await fetch_all("SELECT id FROM rag_queries WHERE user_id=?", (uid,))
    configs = await fetch_all("SELECT id FROM reward_configs WHERE user_id=?", (uid,))

    return {
        "total_domains": len(domains),
        "total_documents": doc_count,
        "total_queries": len(queries),
        "total_reward_configs": len(configs),
        "storage_used_bytes": total_size,
        "storage_limit_bytes": int(settings.max_upload_mb * 1024 * 1024),
    }


# --- API Keys ---

@router.post("/api-keys", response_model=APIKeyCreated)
async def create_key_route(body: APIKeyCreate, user: Dict = Depends(get_current_user)):
    return await create_api_key(user["user_id"], body.name)


@router.get("/api-keys", response_model=List[APIKeyOut])
async def list_keys_route(user: Dict = Depends(get_current_user)):
    return await list_api_keys(user["user_id"])


@router.delete("/api-keys/{key_id}")
async def revoke_key_route(key_id: str, user: Dict = Depends(get_current_user)):
    ok = await revoke_api_key(user["user_id"], key_id)
    if not ok:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"ok": True}
