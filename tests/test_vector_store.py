"""Tests for the ChromaDB vector store."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb")

from agent.memory.embeddings import EmbeddingModel  # noqa: E402
from agent.memory.vectors import VectorResult, VectorStore  # noqa: E402


class FakeEmbeddingModel(EmbeddingModel):
    """Deterministic fake embedding model for tests."""

    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self._dim = dim
        # Set _model to non-None so embed() skips _load_model()
        self._model = True  # type: ignore[assignment]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic vectors seeded by text hash."""
        if not texts:
            return []
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % 2**31)
            vec = rng.randn(self._dim).astype(np.float32)
            # Normalize for cosine similarity
            vec = vec / np.linalg.norm(vec)
            vecs.append(vec.tolist())
        return vecs


@pytest.fixture
def fake_model() -> FakeEmbeddingModel:
    """Create a fake embedding model."""
    return FakeEmbeddingModel()


@pytest.fixture
async def store(tmp_path: Any, fake_model: FakeEmbeddingModel) -> VectorStore:
    """Create an initialized VectorStore with fake embeddings."""
    vs = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
        embedding_model=fake_model,
    )
    await vs.initialize()
    return vs


class TestVectorResult:
    """Tests for VectorResult."""

    def test_score_property(self) -> None:
        """score should be 1 - distance."""
        result = VectorResult(id="1", text="hello", distance=0.3)
        assert abs(result.score - 0.7) < 1e-6

    def test_score_clamped_to_zero(self) -> None:
        """score should not go below 0."""
        result = VectorResult(id="1", text="hello", distance=1.5)
        assert result.score == 0.0

    def test_score_perfect_match(self) -> None:
        """score should be 1.0 for distance 0."""
        result = VectorResult(id="1", text="hello", distance=0.0)
        assert result.score == 1.0


class TestVectorStore:
    """Tests for VectorStore."""

    async def test_add_and_count(self, store: VectorStore) -> None:
        """add() should increase the document count."""
        assert await store.count() == 0
        await store.add("hello world")
        assert await store.count() == 1

    async def test_add_returns_id(self, store: VectorStore) -> None:
        """add() should return a document ID."""
        doc_id = await store.add("test document")
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    async def test_add_custom_id(self, store: VectorStore) -> None:
        """add() should accept a custom document ID."""
        doc_id = await store.add("test", doc_id="custom-123")
        assert doc_id == "custom-123"
        assert await store.count() == 1

    async def test_add_batch(self, store: VectorStore) -> None:
        """add_batch() should add multiple documents."""
        ids = await store.add_batch(["doc1", "doc2", "doc3"])
        assert len(ids) == 3
        assert await store.count() == 3

    async def test_add_batch_empty(self, store: VectorStore) -> None:
        """add_batch() with empty list should return empty list."""
        ids = await store.add_batch([])
        assert ids == []
        assert await store.count() == 0

    async def test_add_batch_with_metadata(self, store: VectorStore) -> None:
        """add_batch() should store metadata."""
        metadatas = [{"session_id": "s1"}, {"session_id": "s2"}]
        await store.add_batch(["doc1", "doc2"], metadatas=metadatas)
        results = await store.search("doc1", limit=2)
        assert len(results) > 0
        # At least one result should have session_id metadata
        metas = [r.metadata for r in results]
        session_ids = {m.get("session_id") for m in metas}
        assert session_ids & {"s1", "s2"}

    async def test_search_returns_results(self, store: VectorStore) -> None:
        """search() should return matching documents."""
        await store.add("The sky is blue")
        await store.add("Python is a programming language")
        results = await store.search("blue sky")
        assert len(results) > 0
        assert all(isinstance(r, VectorResult) for r in results)

    async def test_search_respects_limit(self, store: VectorStore) -> None:
        """search() should respect the limit parameter."""
        for i in range(10):
            await store.add(f"document number {i}")
        results = await store.search("document", limit=3)
        assert len(results) == 3

    async def test_search_empty_collection(self, store: VectorStore) -> None:
        """search() on empty collection should return empty list."""
        results = await store.search("anything")
        assert results == []

    async def test_search_with_where_filter(self, store: VectorStore) -> None:
        """search() with where filter should only return matching docs."""
        await store.add("alpha doc", metadata={"type": "alpha"})
        await store.add("beta doc", metadata={"type": "beta"})
        results = await store.search("doc", where={"type": "alpha"})
        assert len(results) == 1
        assert results[0].metadata["type"] == "alpha"

    async def test_delete(self, store: VectorStore) -> None:
        """delete() should remove a document."""
        doc_id = await store.add("to be deleted", doc_id="del-me")
        assert await store.count() == 1
        await store.delete(doc_id)
        assert await store.count() == 0

    async def test_delete_by_session(self, store: VectorStore) -> None:
        """delete_by_session() should remove all docs for a session."""
        await store.add("doc1", metadata={"session_id": "s1"})
        await store.add("doc2", metadata={"session_id": "s1"})
        await store.add("doc3", metadata={"session_id": "s2"})
        assert await store.count() == 3

        await store.delete_by_session("s1")
        assert await store.count() == 1

    async def test_not_initialized_raises(
        self, tmp_path: Any, fake_model: FakeEmbeddingModel
    ) -> None:
        """Operations on uninitialized store should raise RuntimeError."""
        vs = VectorStore(
            persist_dir=str(tmp_path / "uninit"),
            embedding_model=fake_model,
        )
        with pytest.raises(RuntimeError, match="not initialized"):
            await vs.add("test")
        with pytest.raises(RuntimeError, match="not initialized"):
            await vs.search("test")
        with pytest.raises(RuntimeError, match="not initialized"):
            await vs.count()
