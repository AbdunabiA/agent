"""Memory data models.

Defines the Fact dataclass used by the SQLite facts store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Fact:
    """A structured fact/memory.

    Facts are key-value pairs with metadata for relevance ranking.
    Keys use dot-notation namespaces (e.g. "user.name", "preference.language").

    Attributes:
        id: Unique identifier (UUID).
        key: Dot-notation key (e.g. "user.name").
        value: The fact value as a string.
        category: Grouping category ("user", "preference", "project", "system", "general").
        confidence: Confidence score from 0.0 to 1.0, decays over time.
        source: How the fact was learned ("user", "extracted", "inferred").
        created_at: When the fact was first created.
        updated_at: When the fact was last modified.
        accessed_at: When the fact was last used in context.
        access_count: Number of times the fact was retrieved.
    """

    id: str
    key: str
    value: str
    category: str = "general"
    confidence: float = 1.0
    source: str = "user"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
