"""Token usage and cost tracking for LLM calls.

Tracks per-call usage with model name, channel, and timestamp.
Provides aggregated statistics by time period, model, and channel.

Phase 5B: In-memory storage. Could be extended to SQLite persistence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Approximate pricing per 1M tokens (input, output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-opus": (15.0, 75.0),
    "claude-3-haiku": (0.25, 1.25),
    "claude-haiku": (0.25, 1.25),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gemini-pro": (0.50, 1.50),
    "gemini-1.5-pro": (3.50, 10.50),
    "gemini-1.5-flash": (0.075, 0.30),
}

# Default pricing for unknown models
_DEFAULT_PRICING = (2.0, 8.0)


@dataclass
class UsageEntry:
    """A single LLM usage record.

    Attributes:
        model: Model name used for the call.
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        cost: Estimated cost in USD.
        channel: Source channel (cli, telegram, webchat, api).
        session_id: Associated session ID.
        timestamp: When the call was made.
    """

    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    channel: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost based on model pricing.

    Args:
        model: Model name (matched by substring).
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    model_lower = model.lower()
    pricing = _DEFAULT_PRICING

    for pattern, price in MODEL_PRICING.items():
        if pattern in model_lower:
            pricing = price
            break

    input_cost = (input_tokens / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]
    return input_cost + output_cost


class CostTracker:
    """Tracks LLM token usage and estimated costs.

    In-memory storage with aggregation by time period, model, and channel.

    Usage::

        tracker = CostTracker()
        tracker.record("claude-sonnet-4-5", 1000, 500, channel="telegram")
        stats = tracker.get_stats(period="day")
    """

    def __init__(self) -> None:
        self._entries: list[UsageEntry] = []
        self._start_time = time.time()

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        channel: str = "cli",
        session_id: str = "",
    ) -> UsageEntry:
        """Record a usage entry.

        Args:
            model: Model name used.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            channel: Source channel.
            session_id: Session identifier.

        Returns:
            The created UsageEntry.
        """
        cost = _estimate_cost(model, input_tokens, output_tokens)
        entry = UsageEntry(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            channel=channel,
            session_id=session_id,
        )
        self._entries.append(entry)

        logger.debug(
            "cost_recorded",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=f"${cost:.6f}",
        )
        return entry

    def get_total_cost(self) -> float:
        """Get total estimated cost across all entries.

        Returns:
            Total cost in USD.
        """
        return sum(e.cost for e in self._entries)

    def get_stats(self, period: str = "day") -> dict[str, Any]:
        """Get aggregated stats for a time period.

        Args:
            period: Time period — "day", "week", or "month".

        Returns:
            Dict with total_cost, total_tokens, total_calls, period,
            by_time, by_model, by_channel breakdowns.
        """
        now = datetime.now()
        if period == "week":
            cutoff = now - timedelta(days=7)
        elif period == "month":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = now - timedelta(days=1)

        entries = [e for e in self._entries if e.timestamp >= cutoff]

        if not entries:
            return {
                "total_cost": 0.0,
                "total_tokens": {"input": 0, "output": 0},
                "total_calls": 0,
                "period": period,
                "by_time": [],
                "by_model": [],
                "by_channel": [],
            }

        total_input = sum(e.input_tokens for e in entries)
        total_output = sum(e.output_tokens for e in entries)
        total_cost = sum(e.cost for e in entries)

        # Time buckets
        by_time = self._aggregate_by_time(entries, period)

        # By model
        model_costs: dict[str, float] = {}
        for e in entries:
            model_costs[e.model] = model_costs.get(e.model, 0.0) + e.cost
        by_model = [
            {
                "model": model,
                "cost": cost,
                "percentage": (cost / total_cost * 100) if total_cost > 0 else 0,
            }
            for model, cost in sorted(model_costs.items(), key=lambda x: -x[1])
        ]

        # By channel
        channel_costs: dict[str, float] = {}
        for e in entries:
            channel_costs[e.channel] = channel_costs.get(e.channel, 0.0) + e.cost
        by_channel = [
            {"channel": ch, "cost": cost}
            for ch, cost in sorted(channel_costs.items(), key=lambda x: -x[1])
        ]

        return {
            "total_cost": round(total_cost, 6),
            "total_tokens": {"input": total_input, "output": total_output},
            "total_calls": len(entries),
            "period": period,
            "by_time": by_time,
            "by_model": by_model,
            "by_channel": by_channel,
        }

    def _aggregate_by_time(
        self, entries: list[UsageEntry], period: str
    ) -> list[dict[str, Any]]:
        """Group entries into time buckets.

        Args:
            entries: Filtered usage entries.
            period: Time period for bucket sizing.

        Returns:
            List of {time, cost, tokens} dicts.
        """
        buckets: dict[str, dict[str, float]] = {}

        for e in entries:
            if period == "day":
                key = e.timestamp.strftime("%H:00")
            elif period == "week":
                key = e.timestamp.strftime("%a")
            else:
                key = e.timestamp.strftime("%m/%d")

            if key not in buckets:
                buckets[key] = {"cost": 0.0, "tokens": 0}
            buckets[key]["cost"] += e.cost
            buckets[key]["tokens"] += e.input_tokens + e.output_tokens

        return [
            {"time": k, "cost": round(v["cost"], 6), "tokens": int(v["tokens"])}
            for k, v in buckets.items()
        ]
