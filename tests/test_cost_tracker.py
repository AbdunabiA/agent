"""Tests for the cost tracker module."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agent.core.cost_tracker import (
    _DEFAULT_PRICING,
    MODEL_PRICING,
    CostTracker,
    UsageEntry,
    _estimate_cost,
)


class TestEstimateCost:
    """Tests for the _estimate_cost helper."""

    def test_known_model_pricing(self) -> None:
        """Claude Sonnet should use its configured pricing."""
        cost = _estimate_cost("claude-sonnet-4-5", input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(3.0)

    def test_known_model_output_pricing(self) -> None:
        cost = _estimate_cost("claude-sonnet-4-5", input_tokens=0, output_tokens=1_000_000)
        assert cost == pytest.approx(15.0)

    def test_combined_input_output(self) -> None:
        cost = _estimate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
        expected = 2.50 + 10.0  # gpt-4o pricing
        assert cost == pytest.approx(expected)

    def test_unknown_model_uses_default(self) -> None:
        cost = _estimate_cost("totally-unknown-model", input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(_DEFAULT_PRICING[0])

    def test_case_insensitive_matching(self) -> None:
        """Model name matching is case-insensitive via .lower()."""
        cost = _estimate_cost("Claude-Sonnet-4-5", input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(3.0)

    def test_zero_tokens_zero_cost(self) -> None:
        cost = _estimate_cost("claude-sonnet-4-5", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_small_token_count(self) -> None:
        cost = _estimate_cost("claude-sonnet-4-5", input_tokens=100, output_tokens=50)
        expected = (100 / 1_000_000) * 3.0 + (50 / 1_000_000) * 15.0
        assert cost == pytest.approx(expected)


class TestCostTrackerRecord:
    """Tests for recording token usage."""

    def test_record_returns_entry(self) -> None:
        tracker = CostTracker()
        entry = tracker.record("claude-sonnet-4-5", input_tokens=1000, output_tokens=500)
        assert isinstance(entry, UsageEntry)
        assert entry.model == "claude-sonnet-4-5"
        assert entry.input_tokens == 1000
        assert entry.output_tokens == 500
        assert entry.cost > 0

    def test_record_with_channel(self) -> None:
        tracker = CostTracker()
        entry = tracker.record("gpt-4o", input_tokens=100, output_tokens=50, channel="telegram")
        assert entry.channel == "telegram"

    def test_record_with_session_id(self) -> None:
        tracker = CostTracker()
        entry = tracker.record("gpt-4o", input_tokens=100, output_tokens=50, session_id="sess-123")
        assert entry.session_id == "sess-123"

    def test_default_channel_is_cli(self) -> None:
        tracker = CostTracker()
        entry = tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        assert entry.channel == "cli"

    def test_multiple_records_accumulate(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.record("gpt-4o", input_tokens=200, output_tokens=100)
        assert len(tracker._entries) == 2


class TestGetTotalCost:
    """Tests for total cost calculation."""

    def test_empty_tracker_returns_zero(self) -> None:
        tracker = CostTracker()
        assert tracker.get_total_cost() == 0.0

    def test_single_entry_cost(self) -> None:
        tracker = CostTracker()
        entry = tracker.record("claude-sonnet-4-5", input_tokens=1_000_000, output_tokens=0)
        assert tracker.get_total_cost() == pytest.approx(entry.cost)

    def test_multiple_entries_summed(self) -> None:
        tracker = CostTracker()
        e1 = tracker.record("claude-sonnet-4-5", input_tokens=1_000_000, output_tokens=0)
        e2 = tracker.record("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        assert tracker.get_total_cost() == pytest.approx(e1.cost + e2.cost)


class TestGetStats:
    """Tests for aggregated statistics."""

    def test_empty_stats(self) -> None:
        tracker = CostTracker()
        stats = tracker.get_stats(period="day")
        assert stats["total_cost"] == 0.0
        assert stats["total_tokens"]["input"] == 0
        assert stats["total_tokens"]["output"] == 0
        assert stats["total_calls"] == 0
        assert stats["period"] == "day"

    def test_stats_counts_calls(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.record("gpt-4o", input_tokens=200, output_tokens=100)
        stats = tracker.get_stats(period="day")
        assert stats["total_calls"] == 2

    def test_stats_sums_tokens(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.record("gpt-4o", input_tokens=200, output_tokens=100)
        stats = tracker.get_stats(period="day")
        assert stats["total_tokens"]["input"] == 300
        assert stats["total_tokens"]["output"] == 150

    def test_stats_by_model(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
        tracker.record("claude-sonnet-4-5", input_tokens=1000, output_tokens=500)
        stats = tracker.get_stats(period="day")
        models = {m["model"] for m in stats["by_model"]}
        assert "gpt-4o" in models
        assert "claude-sonnet-4-5" in models

    def test_stats_by_channel(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50, channel="cli")
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50, channel="telegram")
        stats = tracker.get_stats(period="day")
        channels = {c["channel"] for c in stats["by_channel"]}
        assert "cli" in channels
        assert "telegram" in channels

    def test_stats_period_filter_week(self) -> None:
        tracker = CostTracker()
        # Add a recent entry
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        # Manually add an old entry (> 7 days ago)
        old_entry = UsageEntry(
            model="gpt-4o",
            input_tokens=999,
            output_tokens=999,
            cost=0.1,
            channel="cli",
            session_id="",
            timestamp=datetime.now() - timedelta(days=10),
        )
        tracker._entries.append(old_entry)

        stats = tracker.get_stats(period="week")
        # Only the recent entry should be counted
        assert stats["total_calls"] == 1
        assert stats["total_tokens"]["input"] == 100

    def test_stats_period_filter_month(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        old_entry = UsageEntry(
            model="gpt-4o",
            input_tokens=999,
            output_tokens=999,
            cost=0.1,
            channel="cli",
            session_id="",
            timestamp=datetime.now() - timedelta(days=60),
        )
        tracker._entries.append(old_entry)

        stats = tracker.get_stats(period="month")
        assert stats["total_calls"] == 1

    def test_stats_includes_time_buckets(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        stats = tracker.get_stats(period="day")
        assert len(stats["by_time"]) >= 1
        bucket = stats["by_time"][0]
        assert "time" in bucket
        assert "cost" in bucket
        assert "tokens" in bucket


class TestResetTracking:
    """Tests for resetting/clearing the tracker."""

    def test_entries_can_be_cleared(self) -> None:
        """The tracker stores entries in _entries which can be cleared."""
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.record("gpt-4o", input_tokens=200, output_tokens=100)
        assert tracker.get_total_cost() > 0

        # Clear entries directly (no reset method, but entries list is accessible)
        tracker._entries.clear()
        assert tracker.get_total_cost() == 0.0
        assert tracker.get_stats()["total_calls"] == 0

    def test_new_tracker_is_empty(self) -> None:
        """Creating a new CostTracker is equivalent to a reset."""
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=1_000_000, output_tokens=500_000)
        assert tracker.get_total_cost() > 0

        fresh_tracker = CostTracker()
        assert fresh_tracker.get_total_cost() == 0.0
        assert len(fresh_tracker._entries) == 0


class TestModelPricingTable:
    """Tests for the pricing configuration."""

    def test_all_pricing_entries_have_two_values(self) -> None:
        for model, pricing in MODEL_PRICING.items():
            assert len(pricing) == 2, f"Model {model} pricing should be (input, output)"
            assert pricing[0] >= 0, f"Model {model} input price should be non-negative"
            assert pricing[1] >= 0, f"Model {model} output price should be non-negative"

    def test_default_pricing_has_two_values(self) -> None:
        assert len(_DEFAULT_PRICING) == 2
        assert _DEFAULT_PRICING[0] > 0
        assert _DEFAULT_PRICING[1] > 0
