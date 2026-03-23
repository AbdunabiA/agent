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
    # Emotional/contextual metadata (extracted from conversations)
    tone: str = ""  # positive, neutral, negative, urgent
    emotion: str = ""  # comma-separated: excited, concerned, frustrated, grateful
    priority: str = "normal"  # high, normal, low
    topic: str = ""  # topic cluster: deployment, design, personal, etc.
    context_snippet: str = ""  # brief surrounding conversation context
    temporal_reference: str | None = None  # ISO datetime or cron for deadlines
    next_action_date: str | None = None  # when to act on this fact
