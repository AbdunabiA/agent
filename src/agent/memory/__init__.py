"""Memory system — Phase 4.

Provides persistent structured facts storage backed by SQLite,
optional semantic vector search backed by ChromaDB,
fact extraction, conversation summarization, memory decay,
and soul.md personality loading.
"""

from agent.memory.database import Database
from agent.memory.decay import MemoryDecay
from agent.memory.extraction import FactExtractor
from agent.memory.models import Fact
from agent.memory.soul import SoulLoader
from agent.memory.store import FactStore
from agent.memory.summarizer import ConversationSummarizer

__all__ = [
    "ConversationSummarizer",
    "Database",
    "Fact",
    "FactExtractor",
    "FactStore",
    "MemoryDecay",
    "SoulLoader",
]

# Optional: ChromaDB + sentence-transformers (requires pip install 'agent-ai[memory]')
try:
    from agent.memory.embeddings import EmbeddingModel, get_embedding_model
    from agent.memory.vectors import VectorResult, VectorStore

    __all__ += ["EmbeddingModel", "get_embedding_model", "VectorResult", "VectorStore"]
except ImportError:
    pass
