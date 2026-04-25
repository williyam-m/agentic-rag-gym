"""RAG pipeline service — wraps rag_master for multi-tenant usage."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_rag_os.config import get_os_settings

# Regex for basic input sanitization
_SANITIZE_RE = re.compile(r'<[^>]+>|[\x00-\x08\x0b\x0c\x0e-\x1f]')

ALLOWED_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json"}
MAX_CHUNK_CHARS = 1000
CHUNK_OVERLAP = 100


def sanitize_text(text: str) -> str:
    return _SANITIZE_RE.sub("", text).strip()


def _chunk_text(text: str, chunk_size: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks


def _project_index_path(user_id: int, project_id: int) -> Path:
    settings = get_os_settings()
    path = settings.faiss_base_dir / "os_projects" / f"u{user_id}" / f"p{project_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _upload_path(user_id: int, project_id: int) -> Path:
    settings = get_os_settings()
    path = settings.upload_dir / f"u{user_id}" / f"p{project_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class RAGService:
    """Per-project RAG service backed by rag_master FAISSRetriever."""

    def __init__(self, user_id: int, project_id: int) -> None:
        self._user_id = user_id
        self._project_id = project_id
        self._index_path = _project_index_path(user_id, project_id)
        self._retriever: Optional[Any] = None

    async def _get_retriever(self):
        """Lazy-load FAISS retriever."""
        if self._retriever is None:
            try:
                from rag_master.retriever import FAISSRetriever
                settings = get_os_settings()
                self._retriever = FAISSRetriever(
                    index_dir=self._index_path,
                    embedding_model=settings.embedding_model,
                    dimension=settings.faiss_dimension,
                )
            except ImportError:
                raise RuntimeError("rag_master not available")
        return self._retriever

    async def ingest_file(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Ingest a text file into the RAG index. Returns stats."""
        from rag_master.models import Document

        # Validate extension
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"File type '{ext}' not supported. Allowed: {ALLOWED_EXTENSIONS}")

        # Decode and sanitize
        text = sanitize_text(content.decode("utf-8", errors="replace"))
        if not text:
            raise ValueError("File is empty after sanitization.")

        # Save raw file
        safe_name = hashlib.sha256(filename.encode()).hexdigest()[:12] + ext
        upload_path = _upload_path(self._user_id, self._project_id) / safe_name
        upload_path.write_bytes(content)

        # Chunk and embed
        chunks = _chunk_text(text)
        docs = [
            Document(
                content=chunk,
                metadata={"filename": filename, "chunk_index": i},
                source=filename,
            )
            for i, chunk in enumerate(chunks)
        ]
        retriever = await self._get_retriever()
        count = await retriever.index_documents(docs)

        return {
            "filename": filename,
            "size_bytes": len(content),
            "chunks_indexed": count,
            "storage_path": str(upload_path),
        }

    async def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the RAG index. Returns ranked document chunks."""
        q = sanitize_text(query_text)
        if not q:
            return []
        retriever = await self._get_retriever()
        results = await retriever.retrieve(q, top_k=top_k)
        return [
            {
                "content": r.document.content,
                "score": round(r.score, 4),
                "rank": r.rank,
                "source": r.document.source,
                "metadata": r.document.metadata,
            }
            for r in results
        ]

    async def clear(self) -> None:
        """Clear the project index and uploaded files."""
        retriever = await self._get_retriever()
        await retriever.clear_index()
        upload_path = _upload_path(self._user_id, self._project_id)
        if upload_path.exists():
            shutil.rmtree(upload_path)
        self._retriever = None


def get_rag_service(user_id: int, project_id: int) -> RAGService:
    return RAGService(user_id, project_id)
