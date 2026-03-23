"""ChromaDB vector store for semantic memory search.

Stores conversation chunks and retrieves them by semantic similarity
using local embeddings (all-MiniLM-L6-v2).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import structlog

from agent.memory.embeddings import EmbeddingModel, get_embedding_model

logger = structlog.get_logger(__name__)


@dataclass
class VectorResult:
    """A single result from a vector similarity search.

    Attributes:
        id: Document identifier.
        text: The stored text content.
        metadata: Associated metadata dict.
        distance: Cosine distance from the query (0 = identical).
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0

    @property
    def score(self) -> float:
        """Similarity score (1 = identical, 0 = unrelated).

        Clamped to [0, 1] range.
        """
        return max(0.0, 1.0 - self.distance)


class LocalEmbeddingFunction:
    """Bridges our EmbeddingModel to ChromaDB's EmbeddingFunction protocol.

    Implements ``__call__``, ``embed_query``, ``name``, and ``is_legacy``
    as required by ChromaDB >= 1.5.
    """

    def __init__(self, model: EmbeddingModel) -> None:
        self._model = model

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Embed a batch of texts.

        Args:
            input: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self._model.embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Embed a query (same as __call__ for symmetric models).

        Args:
            input: List of query texts.

        Returns:
            List of embedding vectors.
        """
        return self._model.embed(input)

    @staticmethod
    def name() -> str:
        """Return embedding function name (required by ChromaDB protocol)."""
        return "local_embedding"

    @staticmethod
    def is_legacy() -> bool:
        """Signal to ChromaDB that this is a legacy embedding function."""
        return True


class VectorStore:
    """ChromaDB-backed vector store for semantic search.

    Usage::

        store = VectorStore(persist_dir="data/memory")
        await store.initialize()
        doc_id = await store.add("User prefers dark mode", {"session": "abc"})
        results = await store.search("dark theme preference")
    """

    def __init__(
        self,
        persist_dir: str = "data/memory/chroma",
        collection_name: str = "conversations",
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding_model = embedding_model or get_embedding_model()
        self._client: Any = None
        self._collection: Any = None

    async def initialize(self) -> None:
        """Create ChromaDB client and collection.

        Raises:
            ImportError: If chromadb is not installed.
        """
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for vector memory. "
                "Install it with: pip install 'agent-ai[memory]'"
            ) from exc

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        embedding_fn = LocalEmbeddingFunction(self._embedding_model)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "vector_store_initialized",
            persist_dir=self._persist_dir,
            collection=self._collection_name,
        )

    def _ensure_initialized(self) -> None:
        """Guard: raise if initialize() hasn't been called."""
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

    async def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Add a single document to the vector store.

        Args:
            text: The text content to store and index.
            metadata: Optional metadata dict.
            doc_id: Optional custom ID. Generated if not provided.

        Returns:
            The document ID.
        """
        self._ensure_initialized()
        doc_id = doc_id or str(uuid4())
        add_kwargs: dict[str, Any] = {
            "documents": [text],
            "ids": [doc_id],
        }
        if metadata:
            add_kwargs["metadatas"] = [metadata]
        await asyncio.to_thread(self._collection.add, **add_kwargs)
        logger.debug("vector_added", doc_id=doc_id)
        return doc_id

    async def add_batch(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add multiple documents in a single batch.

        Args:
            texts: List of text contents.
            metadatas: Optional list of metadata dicts (same length as texts).
            ids: Optional list of IDs (same length as texts).

        Returns:
            List of document IDs.
        """
        self._ensure_initialized()
        if not texts:
            return []

        doc_ids = ids or [str(uuid4()) for _ in texts]

        add_kwargs: dict[str, Any] = {
            "documents": texts,
            "ids": doc_ids,
        }
        if metadatas:
            add_kwargs["metadatas"] = metadatas
        await asyncio.to_thread(self._collection.add, **add_kwargs)
        logger.debug("vector_batch_added", count=len(texts))
        return doc_ids

    async def search(
        self,
        query: str,
        limit: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorResult]:
        """Search for similar documents by semantic similarity.

        Args:
            query: The search query text.
            limit: Maximum number of results.
            where: Optional ChromaDB where filter for metadata.

        Returns:
            List of VectorResult sorted by similarity (best first).
        """
        self._ensure_initialized()

        # ChromaDB raises if n_results > collection count
        count = await asyncio.to_thread(self._collection.count)
        if count == 0:
            return []
        n_results = min(limit, count)

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        results = await asyncio.to_thread(self._collection.query, **kwargs)

        vector_results: list[VectorResult] = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                vector_results.append(
                    VectorResult(
                        id=doc_id,
                        text=results["documents"][0][i],
                        metadata=(results["metadatas"][0][i] or {}) if results["metadatas"] else {},
                        distance=results["distances"][0][i] if results["distances"] else 0.0,
                    )
                )

        return vector_results

    async def delete(self, doc_id: str) -> None:
        """Delete a document by ID.

        Args:
            doc_id: The document ID to delete.
        """
        self._ensure_initialized()
        await asyncio.to_thread(self._collection.delete, ids=[doc_id])
        logger.debug("vector_deleted", doc_id=doc_id)

    async def delete_by_session(self, session_id: str) -> None:
        """Delete all documents belonging to a session.

        Args:
            session_id: The session ID to filter by.
        """
        self._ensure_initialized()
        await asyncio.to_thread(self._collection.delete, where={"session_id": session_id})
        logger.debug("vectors_deleted_by_session", session_id=session_id)

    async def count(self) -> int:
        """Return the number of documents in the collection.

        Returns:
            Document count.
        """
        self._ensure_initialized()
        return await asyncio.to_thread(self._collection.count)
