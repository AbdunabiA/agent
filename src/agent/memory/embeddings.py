"""Local embedding model for semantic memory.

Wraps sentence-transformers (all-MiniLM-L6-v2) with lazy loading
to avoid penalizing CLI commands that don't need embeddings.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384


class EmbeddingModel:
    """Local embedding model using sentence-transformers.

    The model is lazily loaded on first ``embed()`` call to avoid
    the ~80 MB load penalty for commands that don't need embeddings.

    Usage::

        model = EmbeddingModel()
        vectors = model.embed(["hello world", "foo bar"])
        single = model.embed_single("hello")
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self) -> None:
        """Load the sentence-transformers model (deferred import)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install it with: pip install 'agent-ai[memory]'"
            ) from exc

        logger.info("loading_embedding_model", model=self._model_name)
        self._model = SentenceTransformer(self._model_name)
        logger.info("embedding_model_loaded", model=self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if not texts:
            return []

        if self._model is None:
            self._load_model()

        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return _EMBEDDING_DIM


_singleton: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the module-level singleton EmbeddingModel.

    Returns:
        The shared EmbeddingModel instance.
    """
    global _singleton
    if _singleton is None:
        _singleton = EmbeddingModel()
    return _singleton
