"""Tests documenting the missing credit backpressure in TelemetryAsyncHub."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import pytest

from gym_gui.services.trainer import streams


@dataclass
class _DummyBus:
    published: List[streams.TelemetryEvent] = field(default_factory=list)

    def publish(self, event: streams.TelemetryEvent) -> None:
        self.published.append(event)


@dataclass
class _DummyCreditManager:
    initialize_calls: List[Tuple[str, str]] = field(default_factory=list)
    consume_calls: List[Tuple[str, str]] = field(default_factory=list)
    consume_return: bool = True

    def initialize_stream(self, run_id: str, agent_id: str) -> None:
        self.initialize_calls.append((run_id, agent_id))

    def consume_credit(self, run_id: str, agent_id: str) -> bool:  # pragma: no cover - behaviour under test
        self.consume_calls.append((run_id, agent_id))
        return self.consume_return

    def grant_credits(self, run_id: str, agent_id: str, amount: int) -> None:  # noqa: D401 - required for interface
        """Grant credits (noop for the test double)."""
        # Intentionally left blank; TelemetryAsyncHub never calls this in the current flow.


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=timeout)


def _configure_hub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_queue: int = 16,
    buffer_size: int = 16,
) -> tuple[streams.TelemetryAsyncHub, _DummyBus, _DummyCreditManager, asyncio.Queue]:
    hub = streams.TelemetryAsyncHub(max_queue=max_queue, buffer_size=buffer_size)

    dummy_bus = _DummyBus()
    monkeypatch.setattr(streams, "get_bus", lambda: dummy_bus)

    dummy_credit = _DummyCreditManager()
    hub._credit_mgr = dummy_credit  # type: ignore[attr-defined]

    # Avoid protobuf dependency by returning payload as-is.
    monkeypatch.setattr(streams, "_proto_to_dict", lambda payload: payload)

    queue: asyncio.Queue = asyncio.Queue()
    hub._queue = queue
    return hub, dummy_bus, dummy_credit, queue


@pytest.mark.asyncio
async def test_drain_loop_consumes_credit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single event consumes credit and emits no control event while credits remain."""

    hub, dummy_bus, dummy_credit, queue = _configure_hub(monkeypatch, max_queue=4, buffer_size=4)

    payload = {"agent_id": "agent-1", "timestamp": "2025-10-24T12:00:00Z"}
    await queue.put(("run-1", "step", payload))

    drain_task = asyncio.create_task(hub._drain_loop())
    await _wait_until(lambda: bool(dummy_bus.published))
    hub._stopping = True
    drain_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await drain_task

    assert dummy_bus.published, "Expected the telemetry event to be published"
    assert dummy_credit.initialize_calls == [("run-1", "agent-1")]
    assert dummy_credit.consume_calls == [("run-1", "agent-1")]

    topics = [event.topic for event in dummy_bus.published]
    assert topics == [streams.Topic.STEP_APPENDED], "Unexpected CONTROL event when credits available"


@pytest.mark.asyncio
async def test_fast_publish_emits_control_on_starvation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Burst of telemetry emits control events when credits are exhausted."""

    hub, dummy_bus, dummy_credit, queue = _configure_hub(monkeypatch, max_queue=64, buffer_size=64)
    dummy_credit.consume_return = False  # Simulate credit exhaustion path

    drain_task = asyncio.create_task(hub._drain_loop())

    payload = {"agent_id": "stress-agent", "timestamp": "2025-10-24T12:30:00Z"}
    for step_index in range(48):
        step_payload = dict(payload, step_index=step_index)
        await queue.put(("stress-run", "step", step_payload))

    await _wait_until(lambda: len(dummy_bus.published) >= 48, timeout=2.0)
    hub._stopping = True
    drain_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await drain_task

    control_events = [evt for evt in dummy_bus.published if evt.topic is streams.Topic.CONTROL]
    step_events = [evt for evt in dummy_bus.published if evt.topic is streams.Topic.STEP_APPENDED]

    assert len(step_events) == 48
    assert len(control_events) == 1
    assert control_events[0].payload.get("state") == "STARVED"
    assert dummy_credit.consume_calls == [("stress-run", "stress-agent")] * 48
