"""Tests for the local embedding model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.memory.embeddings import EmbeddingModel, get_embedding_model


def _make_mock_st_class(dim: int = 384) -> MagicMock:
    """Create a mock SentenceTransformer that returns deterministic vectors."""
    mock_model = MagicMock()

    def fake_encode(texts: list[str], **kwargs: object) -> np.ndarray:
        # Deterministic vectors seeded by text hash
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % 2**31)
            vecs.append(rng.randn(dim).astype(np.float32))
        return np.array(vecs)

    mock_model.encode = fake_encode
    return mock_model


class TestEmbeddingModel:
    """Tests for EmbeddingModel."""

    def test_model_is_none_before_first_embed(self) -> None:
        """Model should not be loaded until first embed() call."""
        model = EmbeddingModel()
        assert model._model is None

    @patch("agent.memory.embeddings.logger")
    def test_embed_returns_correct_dimensions(self, _mock_logger: MagicMock) -> None:
        """embed() should return vectors of the correct dimension."""
        model = EmbeddingModel()
        mock_st = _make_mock_st_class(384)

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            # Directly set the model to skip real import
            model._model = mock_st

            result = model.embed(["hello world", "test text"])

        assert len(result) == 2
        assert len(result[0]) == 384
        assert len(result[1]) == 384
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_single_returns_single_vector(self) -> None:
        """embed_single() should return a single vector."""
        model = EmbeddingModel()
        model._model = _make_mock_st_class(384)

        result = model.embed_single("hello")

        assert len(result) == 384
        assert isinstance(result, list)

    def test_embed_empty_returns_empty_without_loading(self) -> None:
        """embed([]) should return [] without loading the model."""
        model = EmbeddingModel()

        result = model.embed([])

        assert result == []
        assert model._model is None  # Model should NOT have been loaded

    def test_dimension_property(self) -> None:
        """dimension property should return 384."""
        model = EmbeddingModel()
        assert model.dimension == 384

    def test_import_error_with_instructions(self) -> None:
        """Should raise ImportError with install instructions."""
        model = EmbeddingModel()

        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            pytest.raises(ImportError, match="pip install"),
        ):
            model._load_model()

    def test_singleton_returns_same_instance(self) -> None:
        """get_embedding_model() should return the same instance."""
        import agent.memory.embeddings as mod

        # Reset singleton
        mod._singleton = None
        try:
            m1 = get_embedding_model()
            m2 = get_embedding_model()
            assert m1 is m2
        finally:
            mod._singleton = None


class TestEmbeddingModelIntegration:
    """Integration tests — skipped if sentence-transformers not installed."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_st(self) -> None:
        pytest.importorskip("sentence_transformers")

    def test_real_embeddings_similar_texts(self) -> None:
        """Similar texts should have cosine similarity > 0.8."""
        model = EmbeddingModel()
        vecs = model.embed(["I love dogs", "I adore puppies"])
        a, b = np.array(vecs[0]), np.array(vecs[1])
        cosine_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cosine_sim > 0.6
