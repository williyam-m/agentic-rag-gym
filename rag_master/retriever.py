"""FAISS-based vector retriever implementation."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_master.adapters import BaseRetriever
from rag_master.logging_config import get_logger
from rag_master.models import Document, RetrievalResult

logger = get_logger(__name__)


class FAISSRetriever(BaseRetriever):
    """FAISS-backed document retriever with sentence-transformer embeddings."""

    def __init__(
        self,
        index_dir: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
    ) -> None:
        self._index_dir = Path(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._dimension = dimension
        self._encoder = SentenceTransformer(embedding_model)
        self._index: Optional[faiss.IndexFlatIP] = None
        self._documents: Dict[int, Document] = {}
        self._next_id = 0
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        """Load existing FAISS index or create a new one."""
        index_path = self._index_dir / "index.faiss"
        docs_path = self._index_dir / "documents.pkl"
        if index_path.exists() and docs_path.exists():
            try:
                self._index = faiss.read_index(str(index_path))
                with open(docs_path, "rb") as f:
                    self._documents = pickle.load(f)  # noqa: S301
                self._next_id = max(self._documents.keys(), default=-1) + 1
                logger.info("faiss_index_loaded", count=len(self._documents))
            except Exception as exc:
                logger.warning("faiss_index_load_failed", error=str(exc))
                self._create_fresh_index()
        else:
            self._create_fresh_index()

    def _create_fresh_index(self) -> None:
        """Create a fresh FAISS index."""
        self._index = faiss.IndexFlatIP(self._dimension)
        self._documents = {}
        self._next_id = 0

    def _save_index(self) -> None:
        """Persist index and documents to disk."""
        if self._index is not None:
            faiss.write_index(self._index, str(self._index_dir / "index.faiss"))
        with open(self._index_dir / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to normalized embeddings."""
        embeddings = self._encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (embeddings / norms).astype(np.float32)

    async def index_documents(self, documents: List[Document]) -> int:
        """Index documents into FAISS. Returns count indexed."""
        if not documents:
            return 0
        texts = [doc.content for doc in documents]
        embeddings = self._encode(texts)

        for i, doc in enumerate(documents):
            idx = self._next_id
            self._documents[idx] = doc
            self._next_id += 1

        assert self._index is not None
        self._index.add(embeddings)
        self._save_index()
        logger.info("documents_indexed", count=len(documents))
        return len(documents)

    async def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k documents for a query."""
        if self._index is None or self._index.ntotal == 0:
            return []

        query_vec = self._encode([query])
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        results: List[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            doc = self._documents.get(int(idx))
            if doc is None:
                continue
            clamped_score = float(max(0.0, min(1.0, (score + 1.0) / 2.0)))
            results.append(
                RetrievalResult(document=doc, score=clamped_score, rank=rank)
            )
        return results

    async def clear_index(self) -> None:
        """Clear all indexed documents."""
        self._create_fresh_index()
        self._save_index()
        logger.info("faiss_index_cleared")
