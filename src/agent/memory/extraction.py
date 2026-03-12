"""Fact extraction pipeline.

Extracts structured facts from conversation messages using LLM
and stores them in the FactStore.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agent.core.session import Session
    from agent.llm.provider import LLMProvider
    from agent.memory.models import Fact
    from agent.memory.store import FactStore

logger = structlog.get_logger(__name__)

EXTRACTION_PROMPT = """\
You are a fact extraction system. Analyze the conversation below and extract \
structured facts about the user or their preferences.

Return a JSON array of objects with these fields:
- "key": dot-notation key (e.g. "user.name", "preference.language", "project.tech_stack")
- "value": the fact value as a string
- "category": one of "user", "preference", "project", "system", "general"

Rules:
- Only extract facts that are clearly stated or strongly implied
- Use dot-notation keys for namespacing (e.g. "user.name", "user.location")
- Keep values concise but complete
- If no facts can be extracted, return an empty array: []

CONVERSATION:
{messages}

Return ONLY a JSON array, no other text:
"""


class FactExtractor:
    """Extracts structured facts from conversations using LLM.

    Processes conversation messages through an LLM to identify
    user facts, preferences, and other structured information,
    then stores them in the FactStore.

    Usage::

        extractor = FactExtractor(llm, fact_store)
        facts = await extractor.extract_from_session(session)
    """

    def __init__(
        self,
        llm: LLMProvider,
        fact_store: FactStore,
        enabled: bool = True,
    ) -> None:
        self.llm = llm
        self.fact_store = fact_store
        self.enabled = enabled

    async def extract_from_messages(
        self, messages: list[dict[str, str]]
    ) -> list[Fact]:
        """Extract facts from a list of messages.

        Sends the messages to the LLM with the extraction prompt,
        parses the JSON response, and stores each fact.

        Args:
            messages: List of message dicts with "role" and "content" keys.

        Returns:
            List of stored Fact objects. Empty list on failure or if disabled.
        """
        if not self.enabled or not messages:
            return []

        # Format messages for the prompt
        formatted = "\n".join(
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages
        )

        prompt_text = EXTRACTION_PROMPT.format(messages=formatted)

        try:
            response = await self.llm.completion(
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.2,
                max_tokens=1024,
            )
        except Exception as e:
            logger.warning("fact_extraction_llm_failed", error=str(e))
            return []

        raw_facts = _parse_facts_json(response.content)
        if not raw_facts:
            return []

        stored_facts: list[Fact] = []
        for item in raw_facts:
            key = item.get("key", "").strip()
            value = item.get("value", "").strip()
            category = item.get("category", "general").strip()

            if not key or not value:
                continue

            try:
                fact = await self.fact_store.set(
                    key=key,
                    value=value,
                    category=category,
                    source="extracted",
                    confidence=0.8,
                )
                stored_facts.append(fact)
            except Exception as e:
                logger.warning("fact_store_failed", key=key, error=str(e))

        if stored_facts:
            logger.info("facts_extracted", count=len(stored_facts))

        return stored_facts

    async def extract_from_session(self, session: Session) -> list[Fact]:
        """Extract facts from the last user+assistant message pair in a session.

        Only processes the last 2 messages to avoid re-extracting old facts.

        Args:
            session: The conversation session.

        Returns:
            List of extracted Fact objects.
        """
        if not self.enabled:
            return []

        # Get last 2 messages (user + assistant pair)
        recent = session.messages[-2:] if len(session.messages) >= 2 else session.messages
        if not recent:
            return []

        from agent.core.session import content_as_text

        messages = [
            {"role": msg.role, "content": content_as_text(msg.content)}
            for msg in recent
            if msg.role in ("user", "assistant")
        ]

        return await self.extract_from_messages(messages)


def _parse_facts_json(text: str) -> list[dict[str, str]]:
    """Parse JSON array from LLM response, handling common formatting issues.

    Strips markdown code fences and handles malformed JSON gracefully.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed list of fact dicts, or empty list on failure.
    """
    if not text or not text.strip():
        return []

    cleaned = text.strip()

    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Handle empty array responses (LLM sometimes returns text instead of [])
    if not cleaned or cleaned == "[]":
        return []

    # Detect common "no facts" text responses before trying JSON parse
    lower = cleaned.lower()
    no_facts_phrases = ("no facts", "no extractable", "cannot extract", "empty array", "nothing to")
    if not cleaned.startswith("[") and any(phrase in lower for phrase in no_facts_phrases):
        return []

    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        # Try to find a JSON array in the text
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        logger.warning(
            "fact_extraction_parse_failed",
            raw_length=len(text),
            raw_preview=text[:200],
        )
        return []
