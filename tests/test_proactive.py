"""Tests for proactive messaging, trigger matching, monitoring, and natural language scheduling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.channels.base import OutgoingMessage
from agent.core.events import EventBus, Events
from agent.core.monitors import Monitor, MonitorManager, MonitorType
from agent.core.proactive import ProactiveMessenger
from agent.core.scheduler import parse_natural_schedule
from agent.skills.base import SkillMetadata
from agent.skills.triggers import TriggerMatcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(
    name: str = "telegram",
    known_user_ids: list[str] | None = None,
) -> MagicMock:
    """Create a mock channel with an async send_message method."""
    channel = MagicMock()
    channel.name = name
    channel.send_message = AsyncMock()
    channel.get_known_user_ids = AsyncMock(return_value=known_user_ids or [])
    # Remove broadcast so the fallback path is exercised
    del channel.broadcast
    return channel


# ===========================================================================
# ProactiveMessenger
# ===========================================================================


class TestProactiveMessenger:
    """Tests for ProactiveMessenger."""

    @pytest.fixture
    def bus(self) -> EventBus:
        return EventBus()

    @pytest.fixture
    def telegram(self) -> MagicMock:
        return _make_channel("telegram", known_user_ids=["u1"])

    @pytest.fixture
    def webchat(self) -> MagicMock:
        return _make_channel("webchat", known_user_ids=["u2"])

    # -- send_proactive: explicit channel ---------------------------------

    async def test_send_proactive_with_explicit_channel(
        self, bus: EventBus, telegram: MagicMock, webchat: MagicMock
    ) -> None:
        messenger = ProactiveMessenger(bus, channels=[telegram, webchat])

        result = await messenger.send_proactive("user1", "hello", channel="webchat")

        assert result is True
        webchat.send_message.assert_awaited_once()
        msg: OutgoingMessage = webchat.send_message.call_args[0][0]
        assert msg.content == "hello"
        assert msg.channel_user_id == "user1"
        assert msg.parse_mode == "Markdown"
        telegram.send_message.assert_not_awaited()

    # -- send_proactive: default (first) channel --------------------------

    async def test_send_proactive_default_channel(
        self, bus: EventBus, telegram: MagicMock, webchat: MagicMock
    ) -> None:
        messenger = ProactiveMessenger(bus, channels=[telegram, webchat])

        result = await messenger.send_proactive("user2", "hi there")

        assert result is True
        telegram.send_message.assert_awaited_once()
        webchat.send_message.assert_not_awaited()

    # -- send_proactive: no channels available ----------------------------

    async def test_send_proactive_no_channels(self, bus: EventBus) -> None:
        messenger = ProactiveMessenger(bus, channels=[])

        result = await messenger.send_proactive("user3", "nothing")

        assert result is False

    # -- send_proactive: named channel not found falls back to first ------

    async def test_send_proactive_named_channel_not_found(
        self, bus: EventBus, telegram: MagicMock
    ) -> None:
        messenger = ProactiveMessenger(bus, channels=[telegram])

        result = await messenger.send_proactive("u1", "msg", channel="nonexistent")

        assert result is True
        telegram.send_message.assert_awaited_once()

    # -- send_proactive: channel raises exception -------------------------

    async def test_send_proactive_channel_error(self, bus: EventBus, telegram: MagicMock) -> None:
        telegram.send_message.side_effect = RuntimeError("connection lost")
        messenger = ProactiveMessenger(bus, channels=[telegram])

        result = await messenger.send_proactive("u1", "boom")

        assert result is False

    # -- send_proactive: emits PROACTIVE_MESSAGE event --------------------

    async def test_send_proactive_emits_event(self, bus: EventBus, telegram: MagicMock) -> None:
        emitted: list[dict] = []

        async def handler(data: object) -> None:
            emitted.append(data)  # type: ignore[arg-type]

        bus.on(Events.PROACTIVE_MESSAGE, handler)
        messenger = ProactiveMessenger(bus, channels=[telegram])

        await messenger.send_proactive("u1", "event test")

        assert len(emitted) == 1
        assert emitted[0]["user_id"] == "u1"
        assert emitted[0]["channel"] == "telegram"

    # -- send_to_all_known_users ------------------------------------------

    async def test_broadcast_to_all_channels(
        self, bus: EventBus, telegram: MagicMock, webchat: MagicMock
    ) -> None:
        messenger = ProactiveMessenger(bus, channels=[telegram, webchat])

        count = await messenger.send_to_all_known_users("broadcast msg")

        assert count == 2
        telegram.send_message.assert_awaited_once()
        webchat.send_message.assert_awaited_once()

        msg: OutgoingMessage = telegram.send_message.call_args[0][0]
        assert msg.channel_user_id == "u1"
        assert msg.content == "broadcast msg"

    async def test_broadcast_no_channels(self, bus: EventBus) -> None:
        messenger = ProactiveMessenger(bus, channels=[])

        count = await messenger.send_to_all_known_users("nobody home")

        assert count == 0

    async def test_broadcast_partial_failure(
        self, bus: EventBus, telegram: MagicMock, webchat: MagicMock
    ) -> None:
        telegram.get_known_user_ids.side_effect = RuntimeError("fail")
        messenger = ProactiveMessenger(bus, channels=[telegram, webchat])

        count = await messenger.send_to_all_known_users("partial")

        assert count == 1  # only webchat succeeded

    # -- add_channel ------------------------------------------------------

    async def test_add_channel_dynamically(self, bus: EventBus, telegram: MagicMock) -> None:
        messenger = ProactiveMessenger(bus, channels=[])

        # No channels yet
        assert await messenger.send_proactive("u1", "nope") is False

        messenger.add_channel(telegram)

        assert await messenger.send_proactive("u1", "now it works") is True
        telegram.send_message.assert_awaited_once()

    async def test_add_channel_accessible_by_name(self, bus: EventBus, webchat: MagicMock) -> None:
        messenger = ProactiveMessenger(bus, channels=[])
        messenger.add_channel(webchat)

        result = await messenger.send_proactive("u1", "by name", channel="webchat")

        assert result is True
        webchat.send_message.assert_awaited_once()


# ===========================================================================
# TriggerMatcher
# ===========================================================================


class TestTriggerMatcher:
    """Tests for TriggerMatcher."""

    @pytest.fixture
    def matcher(self) -> TriggerMatcher:
        return TriggerMatcher()

    def _skill(self, name: str, triggers: list[str]) -> SkillMetadata:
        return SkillMetadata(name=name, triggers=triggers)

    # -- basic matching ---------------------------------------------------

    def test_match_single_trigger(self, matcher: TriggerMatcher) -> None:
        skill = self._skill("weather", ["weather", "forecast"])
        matcher.register_skill(skill)

        matches = matcher.match("What is the weather today?")

        assert len(matches) == 1
        assert matches[0][0].name == "weather"
        assert matches[0][1] == "weather"

    def test_match_second_trigger(self, matcher: TriggerMatcher) -> None:
        skill = self._skill("weather", ["weather", "forecast"])
        matcher.register_skill(skill)

        matches = matcher.match("Give me the forecast")

        assert len(matches) == 1
        assert matches[0][1] == "forecast"

    # -- no match ---------------------------------------------------------

    def test_no_match(self, matcher: TriggerMatcher) -> None:
        skill = self._skill("weather", ["weather", "forecast"])
        matcher.register_skill(skill)

        matches = matcher.match("Tell me a joke")

        assert matches == []

    # -- multiple skill matches -------------------------------------------

    def test_multiple_skills_match(self, matcher: TriggerMatcher) -> None:
        matcher.register_skill(self._skill("weather", ["weather"]))
        matcher.register_skill(self._skill("news", ["news"]))

        matches = matcher.match("Show me the weather and news")

        assert len(matches) == 2
        names = {m[0].name for m in matches}
        assert names == {"weather", "news"}

    # -- unregister skill -------------------------------------------------

    def test_unregister_skill(self, matcher: TriggerMatcher) -> None:
        matcher.register_skill(self._skill("weather", ["weather"]))
        matcher.unregister_skill("weather")

        matches = matcher.match("What is the weather?")

        assert matches == []

    def test_unregister_nonexistent_skill(self, matcher: TriggerMatcher) -> None:
        # Should not raise
        matcher.unregister_skill("does_not_exist")

    # -- empty text -------------------------------------------------------

    def test_empty_text(self, matcher: TriggerMatcher) -> None:
        matcher.register_skill(self._skill("weather", ["weather"]))

        assert matcher.match("") == []

    def test_none_like_empty(self, matcher: TriggerMatcher) -> None:
        """Empty string returns no matches."""
        matcher.register_skill(self._skill("weather", ["weather"]))
        assert matcher.match("") == []

    # -- case insensitivity -----------------------------------------------

    def test_case_insensitive_text(self, matcher: TriggerMatcher) -> None:
        matcher.register_skill(self._skill("weather", ["weather"]))

        matches = matcher.match("WEATHER today?")

        assert len(matches) == 1

    def test_case_insensitive_trigger(self, matcher: TriggerMatcher) -> None:
        matcher.register_skill(self._skill("weather", ["Weather"]))

        matches = matcher.match("the weather is nice")

        assert len(matches) == 1

    def test_mixed_case(self, matcher: TriggerMatcher) -> None:
        matcher.register_skill(self._skill("weather", ["WeAtHeR"]))

        matches = matcher.match("wEaThEr report")

        assert len(matches) == 1

    # -- only one match per skill -----------------------------------------

    def test_one_match_per_skill(self, matcher: TriggerMatcher) -> None:
        """Even if both triggers are in the text, only one match per skill."""
        skill = self._skill("weather", ["weather", "forecast"])
        matcher.register_skill(skill)

        matches = matcher.match("weather forecast for tomorrow")

        assert len(matches) == 1

    # -- skill with no triggers is not registered -------------------------

    def test_skill_without_triggers_ignored(self, matcher: TriggerMatcher) -> None:
        skill = self._skill("empty", [])
        matcher.register_skill(skill)

        assert matcher.match("anything") == []

    # -- no skills registered ---------------------------------------------

    def test_no_skills_registered(self, matcher: TriggerMatcher) -> None:
        assert matcher.match("hello") == []


# ===========================================================================
# MonitorManager
# ===========================================================================


class TestMonitorManager:
    """Tests for MonitorManager."""

    @pytest.fixture
    def bus(self) -> EventBus:
        return EventBus()

    @pytest.fixture
    def proactive(self, bus: EventBus) -> MagicMock:
        mock = MagicMock(spec=ProactiveMessenger)
        mock.send_proactive = AsyncMock(return_value=True)
        return mock

    @pytest.fixture
    def manager(self, bus: EventBus, proactive: MagicMock) -> MonitorManager:
        return MonitorManager(bus, proactive)

    # -- add_monitor ------------------------------------------------------

    async def test_add_monitor(self, manager: MonitorManager) -> None:
        monitor = await manager.add_monitor(
            type=MonitorType.FILE_CHANGE,
            target="/tmp/test.txt",
            interval=30,
            channel="telegram",
            user_id="u1",
            description="Watch test.txt",
        )

        assert isinstance(monitor, Monitor)
        assert monitor.type == MonitorType.FILE_CHANGE
        assert monitor.target == "/tmp/test.txt"
        assert monitor.interval == 30
        assert monitor.channel == "telegram"
        assert monitor.user_id == "u1"
        assert monitor.description == "Watch test.txt"
        assert monitor.id is not None

    async def test_add_monitor_default_description(self, manager: MonitorManager) -> None:
        monitor = await manager.add_monitor(
            type=MonitorType.HTTP_ENDPOINT,
            target="https://example.com",
        )

        assert "http_endpoint" in monitor.description
        assert "https://example.com" in monitor.description

    async def test_add_monitor_all_types(self, manager: MonitorManager) -> None:
        for mtype in MonitorType:
            monitor = await manager.add_monitor(type=mtype, target="test")
            assert monitor.type == mtype

    # -- minimum interval enforcement (10 seconds) ------------------------

    async def test_minimum_interval_enforcement(self, manager: MonitorManager) -> None:
        monitor = await manager.add_monitor(
            type=MonitorType.FILE_CHANGE,
            target="/tmp/x",
            interval=1,
        )

        assert monitor.interval == 10

    async def test_interval_at_minimum(self, manager: MonitorManager) -> None:
        monitor = await manager.add_monitor(
            type=MonitorType.FILE_CHANGE,
            target="/tmp/x",
            interval=10,
        )

        assert monitor.interval == 10

    async def test_interval_above_minimum(self, manager: MonitorManager) -> None:
        monitor = await manager.add_monitor(
            type=MonitorType.FILE_CHANGE,
            target="/tmp/x",
            interval=120,
        )

        assert monitor.interval == 120

    async def test_interval_zero_clamped(self, manager: MonitorManager) -> None:
        monitor = await manager.add_monitor(
            type=MonitorType.SHELL_COMMAND,
            target="echo hi",
            interval=0,
        )

        assert monitor.interval == 10

    async def test_interval_negative_clamped(self, manager: MonitorManager) -> None:
        monitor = await manager.add_monitor(
            type=MonitorType.SHELL_COMMAND,
            target="echo hi",
            interval=-5,
        )

        assert monitor.interval == 10

    # -- remove_monitor ---------------------------------------------------

    async def test_remove_monitor(self, manager: MonitorManager) -> None:
        monitor = await manager.add_monitor(
            type=MonitorType.FILE_CHANGE,
            target="/tmp/x",
        )

        result = manager.remove_monitor(monitor.id)

        assert result is True
        assert manager.list_monitors() == []

    async def test_remove_nonexistent_monitor(self, manager: MonitorManager) -> None:
        result = manager.remove_monitor("nonexistent-id")

        assert result is False

    async def test_remove_monitor_idempotent(self, manager: MonitorManager) -> None:
        monitor = await manager.add_monitor(
            type=MonitorType.GIT_REPO,
            target="/repo",
        )

        assert manager.remove_monitor(monitor.id) is True
        assert manager.remove_monitor(monitor.id) is False

    # -- list_monitors ----------------------------------------------------

    async def test_list_monitors_empty(self, manager: MonitorManager) -> None:
        assert manager.list_monitors() == []

    async def test_list_monitors_multiple(self, manager: MonitorManager) -> None:
        await manager.add_monitor(type=MonitorType.FILE_CHANGE, target="/a")
        await manager.add_monitor(type=MonitorType.HTTP_ENDPOINT, target="http://b")
        await manager.add_monitor(type=MonitorType.SHELL_COMMAND, target="echo c")

        monitors = manager.list_monitors()

        assert len(monitors) == 3
        targets = {m["target"] for m in monitors}
        assert targets == {"/a", "http://b", "echo c"}

    async def test_list_monitors_returns_dicts(self, manager: MonitorManager) -> None:
        await manager.add_monitor(
            type=MonitorType.FILE_CHANGE,
            target="/a",
            interval=60,
            channel="telegram",
            user_id="u1",
            description="Watch a",
        )

        monitors = manager.list_monitors()

        assert len(monitors) == 1
        m = monitors[0]
        assert "id" in m
        assert m["type"] == MonitorType.FILE_CHANGE
        assert m["target"] == "/a"
        assert m["interval"] == 60
        assert m["description"] == "Watch a"
        assert m["channel"] == "telegram"
        assert m["user_id"] == "u1"
        assert m["last_check"] is None

    async def test_list_after_remove(self, manager: MonitorManager) -> None:
        m1 = await manager.add_monitor(type=MonitorType.FILE_CHANGE, target="/a")
        m2 = await manager.add_monitor(type=MonitorType.FILE_CHANGE, target="/b")

        manager.remove_monitor(m1.id)

        monitors = manager.list_monitors()
        assert len(monitors) == 1
        assert monitors[0]["id"] == m2.id

    # -- monitor type enum ------------------------------------------------

    def test_monitor_type_values(self) -> None:
        assert MonitorType.FILE_CHANGE == "file_change"
        assert MonitorType.HTTP_ENDPOINT == "http_endpoint"
        assert MonitorType.GIT_REPO == "git_repo"
        assert MonitorType.SHELL_COMMAND == "shell_command"

    # -- _check_monitor and _notify_change (internal) ---------------------

    async def test_check_monitor_detects_change(
        self, manager: MonitorManager, bus: EventBus, proactive: MagicMock
    ) -> None:
        """When state changes, _check_monitor notifies the user."""
        monitor = await manager.add_monitor(
            type=MonitorType.SHELL_COMMAND,
            target="echo test",
            user_id="u1",
            channel="telegram",
        )
        # Set an initial last_state so a change is detected
        monitor.last_state = "old_state"

        with patch.object(manager, "_get_state", new_callable=AsyncMock, return_value="new_state"):
            await manager._check_monitor(monitor.id)

        proactive.send_proactive.assert_awaited_once()
        assert monitor.last_state == "new_state"
        assert monitor.last_check is not None

    async def test_check_monitor_no_change(
        self, manager: MonitorManager, proactive: MagicMock
    ) -> None:
        """When state is unchanged, no notification is sent."""
        monitor = await manager.add_monitor(
            type=MonitorType.SHELL_COMMAND,
            target="echo stable",
            user_id="u1",
        )
        monitor.last_state = "same"

        with patch.object(manager, "_get_state", new_callable=AsyncMock, return_value="same"):
            await manager._check_monitor(monitor.id)

        proactive.send_proactive.assert_not_awaited()

    async def test_check_monitor_first_run_no_notification(
        self, manager: MonitorManager, proactive: MagicMock
    ) -> None:
        """On first check (last_state is empty), no notification is sent."""
        monitor = await manager.add_monitor(
            type=MonitorType.FILE_CHANGE,
            target="/tmp/first",
            user_id="u1",
        )
        assert monitor.last_state == ""

        with patch.object(
            manager, "_get_state", new_callable=AsyncMock, return_value="initial_state"
        ):
            await manager._check_monitor(monitor.id)

        proactive.send_proactive.assert_not_awaited()
        assert monitor.last_state == "initial_state"

    async def test_check_monitor_nonexistent_id(self, manager: MonitorManager) -> None:
        """Checking a non-existent monitor ID should not raise."""
        await manager._check_monitor("ghost")  # Should simply return

    async def test_notify_emits_monitoring_alert_event(
        self, manager: MonitorManager, bus: EventBus
    ) -> None:
        emitted: list[dict] = []

        async def handler(data: object) -> None:
            emitted.append(data)  # type: ignore[arg-type]

        bus.on(Events.MONITORING_ALERT, handler)

        monitor = await manager.add_monitor(
            type=MonitorType.HTTP_ENDPOINT,
            target="http://example.com",
            user_id="u1",
        )
        monitor.last_state = "200 1000"

        with patch.object(manager, "_get_state", new_callable=AsyncMock, return_value="500 0"):
            await manager._check_monitor(monitor.id)

        assert len(emitted) == 1
        assert emitted[0]["monitor_id"] == monitor.id
        assert emitted[0]["type"] == MonitorType.HTTP_ENDPOINT


# ===========================================================================
# parse_natural_schedule
# ===========================================================================


class TestParseNaturalSchedule:
    """Tests for parse_natural_schedule."""

    # -- "every morning at 8am" -------------------------------------------

    def test_every_morning_at_8am(self) -> None:
        result = parse_natural_schedule("every morning at 8am")
        assert result == "0 8 * * *"

    def test_every_morning_at_8_no_suffix(self) -> None:
        result = parse_natural_schedule("every morning at 8")
        assert result == "0 8 * * *"

    # -- "every Friday at 5pm" --------------------------------------------

    def test_every_friday_at_5pm(self) -> None:
        result = parse_natural_schedule("every Friday at 5pm")
        assert result == "0 17 * * 5"

    def test_every_friday_at_5_am(self) -> None:
        result = parse_natural_schedule("every Friday at 5am")
        assert result == "0 5 * * 5"

    def test_every_friday_at_5_no_suffix(self) -> None:
        result = parse_natural_schedule("every Friday at 5")
        assert result == "0 5 * * 5"

    # -- "every 30 minutes" -----------------------------------------------

    def test_every_30_minutes(self) -> None:
        result = parse_natural_schedule("every 30 minutes")
        assert result == "*/30 * * * *"

    def test_every_1_minute(self) -> None:
        result = parse_natural_schedule("every 1 minute")
        assert result == "*/1 * * * *"

    # -- "every day at noon" -> "0 12 * * *" ------------------------------

    def test_every_day_at_noon(self) -> None:
        result = parse_natural_schedule("every day at noon")
        assert result == "0 12 * * *"

    # -- "every hour" -> "0 * * * *" --------------------------------------

    def test_every_hour(self) -> None:
        result = parse_natural_schedule("every hour")
        assert result == "0 * * * *"

    # -- invalid input returns None ---------------------------------------

    def test_invalid_input(self) -> None:
        assert parse_natural_schedule("do something random") is None

    def test_empty_string(self) -> None:
        assert parse_natural_schedule("") is None

    def test_nonsense(self) -> None:
        assert parse_natural_schedule("banana milkshake") is None

    # -- already a cron expression passes through -------------------------

    def test_cron_expression_passthrough(self) -> None:
        result = parse_natural_schedule("*/5 * * * *")
        assert result == "*/5 * * * *"

    def test_cron_expression_full(self) -> None:
        result = parse_natural_schedule("0 8 * * 1")
        assert result == "0 8 * * 1"

    def test_cron_with_ranges(self) -> None:
        result = parse_natural_schedule("0 9-17 * * 1-5")
        assert result == "0 9-17 * * 1-5"

    # -- more natural language patterns -----------------------------------

    def test_every_day_at_midnight(self) -> None:
        result = parse_natural_schedule("every day at midnight")
        assert result == "0 0 * * *"

    def test_daily_at_9am(self) -> None:
        result = parse_natural_schedule("daily at 9am")
        assert result == "0 9 * * *"

    def test_daily_at_3pm(self) -> None:
        result = parse_natural_schedule("daily at 3pm")
        assert result == "0 15 * * *"

    def test_every_2_hours(self) -> None:
        result = parse_natural_schedule("every 2 hours")
        assert result == "0 */2 * * *"

    def test_every_monday_at_9(self) -> None:
        result = parse_natural_schedule("every monday at 9")
        assert result == "0 9 * * 1"

    def test_every_evening_at_6pm(self) -> None:
        result = parse_natural_schedule("every evening at 6pm")
        assert result == "0 18 * * *"

    # -- case insensitivity -----------------------------------------------

    def test_case_insensitive(self) -> None:
        result = parse_natural_schedule("Every Morning At 8am")
        assert result == "0 8 * * *"

    def test_all_caps(self) -> None:
        result = parse_natural_schedule("EVERY HOUR")
        assert result == "0 * * * *"

    # -- whitespace handling ----------------------------------------------

    def test_leading_trailing_whitespace(self) -> None:
        result = parse_natural_schedule("  every hour  ")
        assert result == "0 * * * *"

    # -- not-quite-cron rejected ------------------------------------------

    def test_four_fields_not_cron(self) -> None:
        """Four fields is not a valid cron expression."""
        assert parse_natural_schedule("1 2 3 4") is None

    def test_six_fields_not_cron(self) -> None:
        """Six fields is not a valid cron expression."""
        assert parse_natural_schedule("1 2 3 4 5 6") is None

    def test_five_fields_with_letters_not_cron(self) -> None:
        """Five fields containing letters are not valid cron."""
        assert parse_natural_schedule("a b c d e") is None
