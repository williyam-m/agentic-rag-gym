"""RAG service — user document ingestion, embedding, and retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agentic_rag_os.config import get_settings
from agentic_rag_os.models import execute, fetch_all, fetch_one, new_id, now_iso

# Lazy-loaded retriever per domain
_retrievers: Dict[str, Any] = {}


def _get_retriever(domain_id: str):
    """Get or create a FAISS retriever for a user domain."""
    if domain_id not in _retrievers:
        from rag_master.retriever import FAISSRetriever
        settings = get_settings()
        index_dir = Path(settings.db_path).parent / "faiss" / domain_id
        index_dir.mkdir(parents=True, exist_ok=True)
        _retrievers[domain_id] = FAISSRetriever(
            index_dir=index_dir,
            embedding_model=settings.embedding_model,
            dimension=settings.faiss_dimension,
        )
    return _retrievers[domain_id]


async def create_domain(user_id: str, name: str, description: str = "") -> Dict[str, Any]:
    """Create a new domain for the user."""
    did = new_id()
    ts = now_iso()
    await execute(
        "INSERT INTO domains (id, user_id, name, description, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (did, user_id, name, description, ts, ts),
    )
    return {"id": did, "name": name, "description": description, "document_count": 0, "total_size_bytes": 0, "created_at": ts}


async def list_domains(user_id: str) -> List[Dict[str, Any]]:
    domains = await fetch_all("SELECT * FROM domains WHERE user_id=?", (user_id,))
    result = []
    for d in domains:
        docs = await fetch_all("SELECT id, size_bytes FROM documents WHERE domain_id=?", (d["id"],))
        result.append({
            "id": d["id"],
            "name": d["name"],
            "description": d["description"],
            "document_count": len(docs),
            "total_size_bytes": sum(doc["size_bytes"] for doc in docs),
            "created_at": d["created_at"],
        })
    return result


async def get_domain(user_id: str, domain_id: str) -> Optional[Dict[str, Any]]:
    return await fetch_one("SELECT * FROM domains WHERE id=? AND user_id=?", (domain_id, user_id))


async def delete_domain(user_id: str, domain_id: str) -> bool:
    d = await fetch_one("SELECT id FROM domains WHERE id=? AND user_id=?", (domain_id, user_id))
    if not d:
        return False
    await execute("DELETE FROM documents WHERE domain_id=?", (domain_id,))
    await execute("DELETE FROM rag_queries WHERE domain_id=?", (domain_id,))
    await execute("DELETE FROM domains WHERE id=?", (domain_id,))
    if domain_id in _retrievers:
        del _retrievers[domain_id]
    return True


async def upload_document(user_id: str, domain_id: str, filename: str, content: str) -> Dict[str, Any]:
    """Upload a text document to a domain."""
    settings = get_settings()

    # Verify domain ownership
    domain = await fetch_one("SELECT * FROM domains WHERE id=? AND user_id=?", (domain_id, user_id))
    if not domain:
        raise ValueError("Domain not found")

    # Check storage limits
    docs = await fetch_all("SELECT size_bytes FROM documents WHERE domain_id IN (SELECT id FROM domains WHERE user_id=?)", (user_id,))
    total_used = sum(d["size_bytes"] for d in docs)
    doc_size = len(content.encode("utf-8"))
    limit = int(settings.max_upload_mb * 1024 * 1024)
    if total_used + doc_size > limit:
        raise ValueError(f"Storage limit exceeded. Free tier allows {settings.max_upload_mb}MB. Upgrade to Premium (coming soon) for {settings.premium_upload_mb}MB.")

    doc_id = new_id()
    ts = now_iso()
    await execute(
        "INSERT INTO documents (id, domain_id, filename, content, size_bytes, created_at) VALUES (?,?,?,?,?,?)",
        (doc_id, domain_id, filename, content, doc_size, ts),
    )

    # Index in FAISS
    retriever = _get_retriever(domain_id)
    from rag_master.models import Document
    doc_obj = Document(doc_id=doc_id, content=content, source=filename, metadata={"filename": filename})
    retriever.index_documents([doc_obj])

    return {"id": doc_id, "filename": filename, "size_bytes": doc_size, "metadata": {}, "created_at": ts}


async def list_documents(user_id: str, domain_id: str) -> List[Dict[str, Any]]:
    domain = await fetch_one("SELECT id FROM domains WHERE id=? AND user_id=?", (domain_id, user_id))
    if not domain:
        return []
    docs = await fetch_all("SELECT id, filename, size_bytes, metadata, created_at FROM documents WHERE domain_id=?", (domain_id,))
    return [
        {"id": d["id"], "filename": d["filename"], "size_bytes": d["size_bytes"],
         "metadata": json.loads(d.get("metadata", "{}")), "created_at": d["created_at"]}
        for d in docs
    ]


async def delete_document(user_id: str, domain_id: str, doc_id: str) -> bool:
    domain = await fetch_one("SELECT id FROM domains WHERE id=? AND user_id=?", (domain_id, user_id))
    if not domain:
        return False
    doc = await fetch_one("SELECT id FROM documents WHERE id=? AND domain_id=?", (doc_id, domain_id))
    if not doc:
        return False
    await execute("DELETE FROM documents WHERE id=?", (doc_id,))
    return True


async def rag_query(user_id: str, domain_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
    """Run a RAG query against user's domain documents."""
    domain = await fetch_one("SELECT * FROM domains WHERE id=? AND user_id=?", (domain_id, user_id))
    if not domain:
        raise ValueError("Domain not found")

    # Ensure all docs are indexed
    retriever = _get_retriever(domain_id)

    results = retriever.retrieve(query, top_k=top_k)
    formatted = [
        {"content": r.document.content, "score": round(r.score, 4), "metadata": r.document.metadata}
        for r in results
    ]

    # Log query
    qid = new_id()
    ts = now_iso()
    await execute(
        "INSERT INTO rag_queries (id, user_id, domain_id, query, results, created_at) VALUES (?,?,?,?,?,?)",
        (qid, user_id, domain_id, query, json.dumps(formatted), ts),
    )

    return {"query": query, "results": formatted, "domain": domain["name"]}


async def reindex_domain(domain_id: str) -> None:
    """Reindex all documents in a domain's FAISS index."""
    retriever = _get_retriever(domain_id)
    retriever.clear_index()
    docs = await fetch_all("SELECT id, filename, content FROM documents WHERE domain_id=?", (domain_id,))
    if docs:
        from rag_master.models import Document
        doc_objs = [
            Document(doc_id=d["id"], content=d["content"], source=d["filename"], metadata={"filename": d["filename"]})
            for d in docs
        ]
        retriever.index_documents(doc_objs)
