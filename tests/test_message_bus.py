"""Tests for the inter-agent message bus."""

from __future__ import annotations

import pytest

from agent.core.events import EventBus
from agent.core.message_bus import AgentMessage, MessageBus


@pytest.fixture
def bus(event_bus: EventBus) -> MessageBus:
    """Create a MessageBus with an event bus but no database."""
    return MessageBus(event_bus=event_bus)


@pytest.fixture
def bus_no_events() -> MessageBus:
    """Create a MessageBus with no event bus and no database."""
    return MessageBus()


# --- Send / Receive ---


@pytest.mark.asyncio
async def test_send_and_receive(bus: MessageBus) -> None:
    msg = AgentMessage(
        task_id="t1",
        from_role="dev",
        to_role="qa",
        content="Please review",
        msg_type="question",
    )
    msg_id = await bus.send(msg)
    assert msg_id == msg.id

    messages = await bus.get_messages(task_id="t1", role="qa")
    assert len(messages) == 1
    assert messages[0].content == "Please review"
    assert messages[0].from_role == "dev"


@pytest.mark.asyncio
async def test_get_messages_filters_by_role(bus: MessageBus) -> None:
    await bus.send(AgentMessage(task_id="t1", from_role="dev", to_role="qa", content="for qa"))
    await bus.send(AgentMessage(task_id="t1", from_role="dev", to_role="arch", content="for arch"))

    qa_msgs = await bus.get_messages(task_id="t1", role="qa")
    assert len(qa_msgs) == 1
    assert qa_msgs[0].content == "for qa"

    arch_msgs = await bus.get_messages(task_id="t1", role="arch")
    assert len(arch_msgs) == 1
    assert arch_msgs[0].content == "for arch"


@pytest.mark.asyncio
async def test_get_messages_empty(bus: MessageBus) -> None:
    messages = await bus.get_messages(task_id="nonexistent", role="dev")
    assert messages == []


# --- Broadcast ---


@pytest.mark.asyncio
async def test_broadcast(bus: MessageBus) -> None:
    msg = AgentMessage(
        task_id="t1",
        from_role="lead",
        to_role="ignored",  # broadcast() should set this to None
        content="Team update",
        msg_type="status",
    )
    msg_id = await bus.broadcast(msg)
    assert msg_id == msg.id
    assert msg.to_role is None

    # Any role should see broadcasts
    qa_msgs = await bus.get_messages(task_id="t1", role="qa")
    assert len(qa_msgs) == 1
    assert qa_msgs[0].content == "Team update"

    dev_msgs = await bus.get_messages(task_id="t1", role="dev")
    assert len(dev_msgs) == 1


# --- Threading ---


@pytest.mark.asyncio
async def test_thread_auto_id(bus: MessageBus) -> None:
    """When no thread_id is given, it defaults to the message id."""
    msg = AgentMessage(task_id="t1", from_role="dev", to_role="qa", content="hi")
    await bus.send(msg)
    assert msg.thread_id == msg.id


@pytest.mark.asyncio
async def test_get_thread(bus: MessageBus) -> None:
    msg1 = AgentMessage(task_id="t1", from_role="dev", to_role="qa", content="question")
    await bus.send(msg1)

    msg2 = AgentMessage(
        task_id="t1",
        from_role="qa",
        to_role="dev",
        content="answer",
        thread_id=msg1.thread_id,
        reply_to=msg1.id,
    )
    await bus.send(msg2)

    thread = await bus.get_thread(msg1.thread_id)
    assert len(thread) == 2
    assert thread[0].content == "question"
    assert thread[1].content == "answer"
    assert thread[1].reply_to == msg1.id


@pytest.mark.asyncio
async def test_get_thread_empty(bus: MessageBus) -> None:
    thread = await bus.get_thread("no-such-thread")
    assert thread == []


# --- Unread tracking ---


@pytest.mark.asyncio
async def test_unread_tracking(bus: MessageBus) -> None:
    await bus.send(AgentMessage(task_id="t1", from_role="dev", to_role="qa", content="msg1"))
    await bus.send(AgentMessage(task_id="t1", from_role="dev", to_role="qa", content="msg2"))

    # First read: both unread
    unread = await bus.get_messages(task_id="t1", role="qa", unread_only=True)
    assert len(unread) == 2

    # Second read: now marked as read
    unread = await bus.get_messages(task_id="t1", role="qa", unread_only=True)
    assert len(unread) == 0

    # But reading all (unread_only=False) still returns them
    all_msgs = await bus.get_messages(task_id="t1", role="qa", unread_only=False)
    assert len(all_msgs) == 2


@pytest.mark.asyncio
async def test_unread_per_role(bus: MessageBus) -> None:
    """Read status is tracked per-role."""
    msg = AgentMessage(task_id="t1", from_role="lead", to_role=None, content="broadcast")
    await bus.broadcast(msg)

    # qa reads it
    await bus.get_messages(task_id="t1", role="qa", unread_only=True)

    # dev hasn't read it yet
    dev_unread = await bus.get_messages(task_id="t1", role="dev", unread_only=True)
    assert len(dev_unread) == 1

    # Now dev has read it
    dev_unread2 = await bus.get_messages(task_id="t1", role="dev", unread_only=True)
    assert len(dev_unread2) == 0


# --- Subscribe ---


@pytest.mark.asyncio
async def test_subscribe(bus: MessageBus) -> None:
    received: list[AgentMessage] = []

    async def on_message(msg: AgentMessage) -> None:
        received.append(msg)

    bus.subscribe("qa", on_message)

    await bus.send(AgentMessage(task_id="t1", from_role="dev", to_role="qa", content="hi"))

    assert len(received) == 1
    assert received[0].content == "hi"


@pytest.mark.asyncio
async def test_subscribe_not_called_for_other_roles(bus: MessageBus) -> None:
    received: list[AgentMessage] = []

    async def on_message(msg: AgentMessage) -> None:
        received.append(msg)

    bus.subscribe("qa", on_message)

    await bus.send(AgentMessage(task_id="t1", from_role="dev", to_role="arch", content="hi"))

    assert len(received) == 0


@pytest.mark.asyncio
async def test_broadcast_notifies_all_subscribers(bus: MessageBus) -> None:
    qa_received: list[AgentMessage] = []
    dev_received: list[AgentMessage] = []

    async def on_qa(msg: AgentMessage) -> None:
        qa_received.append(msg)

    async def on_dev(msg: AgentMessage) -> None:
        dev_received.append(msg)

    bus.subscribe("qa", on_qa)
    bus.subscribe("dev", on_dev)

    msg = AgentMessage(task_id="t1", from_role="lead", content="update")
    await bus.broadcast(msg)

    assert len(qa_received) == 1
    assert len(dev_received) == 1


# --- Persistence mock ---


@pytest.mark.asyncio
async def test_persist_called(bus: MessageBus) -> None:
    """Verify _persist is called (no DB, so just ensure no error)."""
    msg = AgentMessage(task_id="t1", from_role="dev", to_role="qa", content="test")
    # No database, so _persist should be a no-op
    await bus.send(msg)
    # No exception means success


@pytest.mark.asyncio
async def test_persist_with_mock_db(event_bus: EventBus) -> None:
    """Verify _persist writes to DB when database is available."""
    from unittest.mock import AsyncMock, MagicMock

    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.fetch_all = AsyncMock(return_value=[])

    bus = MessageBus(event_bus=event_bus, database=mock_db)

    msg = AgentMessage(task_id="t1", from_role="dev", to_role="qa", content="persisted")
    await bus.send(msg)

    mock_db.execute.assert_called_once()
    call_args = mock_db.execute.call_args
    assert "INSERT INTO agent_messages" in call_args[0][0]
    assert call_args[0][1][0] == msg.id


# --- Load from DB ---


@pytest.mark.asyncio
async def test_load_messages_from_db(event_bus: EventBus) -> None:
    """Verify _load_messages populates in-memory store from DB."""
    from unittest.mock import AsyncMock, MagicMock

    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.fetch_all = AsyncMock(
        return_value=[
            ("m1", "t1", "dev", "qa", None, "m1", "hello from db", "question", None, 1000.0),
        ]
    )

    bus = MessageBus(event_bus=event_bus, database=mock_db)

    # get_messages triggers _load_messages
    messages = await bus.get_messages(task_id="t1", role="qa")
    assert len(messages) == 1
    assert messages[0].content == "hello from db"
    assert messages[0].id == "m1"


@pytest.mark.asyncio
async def test_load_messages_only_once(event_bus: EventBus) -> None:
    """Verify _load_messages only queries DB once per task_id."""
    from unittest.mock import AsyncMock, MagicMock

    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.fetch_all = AsyncMock(return_value=[])

    bus = MessageBus(event_bus=event_bus, database=mock_db)

    await bus.get_messages(task_id="t1", role="qa")
    await bus.get_messages(task_id="t1", role="qa")

    # fetch_all should be called only once for t1
    assert mock_db.fetch_all.call_count == 1


# --- No event bus ---


@pytest.mark.asyncio
async def test_send_without_event_bus(bus_no_events: MessageBus) -> None:
    """MessageBus works fine without an event bus."""
    msg = AgentMessage(task_id="t1", from_role="dev", to_role="qa", content="no events")
    msg_id = await bus_no_events.send(msg)
    assert msg_id

    messages = await bus_no_events.get_messages(task_id="t1", role="qa")
    assert len(messages) == 1


# --- Edge cases ---


@pytest.mark.asyncio
async def test_message_to_nonexistent_role(bus: MessageBus) -> None:
    """Sending to a role with no subscribers should still store the message."""
    msg = AgentMessage(
        task_id="t1",
        from_role="dev",
        to_role="nobody",
        content="hello?",
    )
    msg_id = await bus.send(msg)
    assert msg_id == msg.id

    messages = await bus.get_messages(task_id="t1", role="nobody")
    assert len(messages) == 1
    assert messages[0].content == "hello?"


@pytest.mark.asyncio
async def test_empty_content_message(bus: MessageBus) -> None:
    """Empty content messages should still be sent and stored."""
    msg = AgentMessage(task_id="t1", from_role="dev", to_role="qa", content="")
    msg_id = await bus.send(msg)
    assert msg_id

    messages = await bus.get_messages(task_id="t1", role="qa")
    assert len(messages) == 1
    assert messages[0].content == ""


@pytest.mark.asyncio
async def test_very_long_message(bus: MessageBus) -> None:
    """Messages with very long content (10000+ chars) should work."""
    long_content = "x" * 15000
    msg = AgentMessage(
        task_id="t1",
        from_role="dev",
        to_role="qa",
        content=long_content,
    )
    msg_id = await bus.send(msg)
    assert msg_id

    messages = await bus.get_messages(task_id="t1", role="qa")
    assert len(messages) == 1
    assert len(messages[0].content) == 15000


@pytest.mark.asyncio
async def test_reply_chain(bus: MessageBus) -> None:
    """Multi-level reply chain: A->B->A should track thread correctly."""
    msg1 = AgentMessage(
        task_id="t1",
        from_role="dev",
        to_role="qa",
        content="question?",
    )
    await bus.send(msg1)

    msg2 = AgentMessage(
        task_id="t1",
        from_role="qa",
        to_role="dev",
        content="answer!",
        thread_id=msg1.thread_id,
        reply_to=msg1.id,
    )
    await bus.send(msg2)

    msg3 = AgentMessage(
        task_id="t1",
        from_role="dev",
        to_role="qa",
        content="follow-up",
        thread_id=msg1.thread_id,
        reply_to=msg2.id,
    )
    await bus.send(msg3)

    thread = await bus.get_thread(msg1.thread_id)
    assert len(thread) == 3
    assert thread[0].content == "question?"
    assert thread[1].content == "answer!"
    assert thread[1].reply_to == msg1.id
    assert thread[2].content == "follow-up"
    assert thread[2].reply_to == msg2.id
    # All share the same thread_id
    assert all(m.thread_id == msg1.thread_id for m in thread)


@pytest.mark.asyncio
async def test_get_messages_different_task_ids(bus: MessageBus) -> None:
    """Messages from different task_ids should not cross-contaminate."""
    await bus.send(
        AgentMessage(task_id="t1", from_role="dev", to_role="qa", content="task1 msg"),
    )
    await bus.send(
        AgentMessage(task_id="t2", from_role="dev", to_role="qa", content="task2 msg"),
    )

    t1_msgs = await bus.get_messages(task_id="t1", role="qa")
    t2_msgs = await bus.get_messages(task_id="t2", role="qa")

    assert len(t1_msgs) == 1
    assert t1_msgs[0].content == "task1 msg"
    assert len(t2_msgs) == 1
    assert t2_msgs[0].content == "task2 msg"


@pytest.mark.asyncio
async def test_subscriber_exception_doesnt_crash_bus(bus: MessageBus) -> None:
    """If a subscriber callback raises, other subscribers still get notified."""
    received: list[AgentMessage] = []

    async def bad_callback(msg: AgentMessage) -> None:
        raise RuntimeError("subscriber exploded")

    async def good_callback(msg: AgentMessage) -> None:
        received.append(msg)

    bus.subscribe("qa", bad_callback)
    bus.subscribe("qa", good_callback)

    msg = AgentMessage(task_id="t1", from_role="dev", to_role="qa", content="test")
    await bus.send(msg)

    # The good subscriber should still have been called despite the bad one raising
    assert len(received) == 1
    assert received[0].content == "test"


@pytest.mark.asyncio
async def test_broadcast_includes_sender_in_recipients(bus: MessageBus) -> None:
    """Broadcast should be visible to all roles including sender."""
    msg = AgentMessage(
        task_id="t1",
        from_role="lead",
        content="team update",
    )
    await bus.broadcast(msg)

    # Sender can also read the broadcast
    sender_msgs = await bus.get_messages(task_id="t1", role="lead")
    assert len(sender_msgs) == 1
    assert sender_msgs[0].content == "team update"

    # Other roles can too
    other_msgs = await bus.get_messages(task_id="t1", role="dev")
    assert len(other_msgs) == 1
