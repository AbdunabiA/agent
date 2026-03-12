"""Conversation summarizer.

Generates concise summaries of conversation sessions using LLM
and stores them in ChromaDB for semantic retrieval.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agent.core.session import Session
    from agent.llm.provider import LLMProvider
    from agent.memory.vectors import VectorStore

logger = structlog.get_logger(__name__)

SUMMARY_PROMPT = """\
Summarize the following conversation in 2-3 sentences. Focus on:
- Key topics discussed
- Decisions made or actions taken
- Important information shared by the user

CONVERSATION:
{messages}

Summary:
"""


class ConversationSummarizer:
    """Generates conversation summaries and stores them in ChromaDB.

    Summarizes conversations when they reach a configurable length threshold,
    storing the summaries as vector embeddings for later semantic retrieval.

    Usage::

        summarizer = ConversationSummarizer(llm, vector_store)
        summary = await summarizer.summarize_session(session)
        summary = await summarizer.summarize_if_needed(session, threshold=20)
    """

    def __init__(self, llm: LLMProvider, vector_store: VectorStore) -> None:
        self.llm = llm
        self.vector_store = vector_store

    async def summarize_session(self, session: Session) -> str:
        """Generate a summary of the session and store it in ChromaDB.

        Args:
            session: The conversation session to summarize.

        Returns:
            The generated summary text.

        Raises:
            Exception: If LLM call fails.
        """
        if not session.messages:
            return ""

        from agent.core.session import content_as_text

        # Format messages for the prompt
        formatted = "\n".join(
            f"{msg.role}: {content_as_text(msg.content)}"
            for msg in session.messages
            if msg.role in ("user", "assistant") and msg.content
        )

        if not formatted.strip():
            return ""

        prompt_text = SUMMARY_PROMPT.format(messages=formatted)

        response = await self.llm.completion(
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.3,
            max_tokens=512,
        )

        summary = response.content.strip()

        # Store in vector store with metadata
        await self.vector_store.add(
            text=summary,
            metadata={
                "session_id": session.id,
                "type": "summary",
                "message_count": session.message_count,
                "created_at": datetime.now().isoformat(),
            },
        )

        logger.info(
            "session_summarized",
            session_id=session.id,
            message_count=session.message_count,
            summary_length=len(summary),
        )

        return summary

    async def summarize_if_needed(
        self, session: Session, threshold: int = 20
    ) -> str | None:
        """Summarize the session only if it has enough messages.

        Args:
            session: The conversation session.
            threshold: Minimum message count to trigger summarization.

        Returns:
            The summary text if summarized, None if below threshold.
        """
        if session.message_count < threshold:
            return None

        return await self.summarize_session(session)
