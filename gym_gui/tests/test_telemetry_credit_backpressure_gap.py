"""Regression tests documenting missing credit enforcement in TelemetryAsyncHub."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import pytest

from gym_gui.services.trainer import streams


@dataclass
class _RecordingBus:
    published: List[streams.TelemetryEvent] = field(default_factory=list)

    def publish(self, event: streams.TelemetryEvent) -> None:
        self.published.append(event)


@dataclass
class _RecordingCreditManager:
    initialize_calls: List[Tuple[str, str]] = field(default_factory=list)
    consume_calls: List[Tuple[str, str]] = field(default_factory=list)
    consume_return: bool = True

    def initialize_stream(self, run_id: str, agent_id: str) -> None:
        self.initialize_calls.append((run_id, agent_id))

    def consume_credit(self, run_id: str, agent_id: str) -> bool:  # pragma: no cover - behaviour under observation
        self.consume_calls.append((run_id, agent_id))
        return self.consume_return

    def grant_credits(self, run_id: str, agent_id: str, amount: int) -> None:
        # No-op for the test double
        return


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=timeout)


def _configure_hub(monkeypatch: pytest.MonkeyPatch) -> tuple[streams.TelemetryAsyncHub, _RecordingBus, _RecordingCreditManager, asyncio.Queue]:
    hub = streams.TelemetryAsyncHub(max_queue=32, buffer_size=32)

    bus = _RecordingBus()
    monkeypatch.setattr(streams, "get_bus", lambda: bus)

    credit_mgr = _RecordingCreditManager()
    hub._credit_mgr = credit_mgr  # type: ignore[attr-defined]

    monkeypatch.setattr(streams, "_proto_to_dict", lambda payload: payload)

    queue: asyncio.Queue = asyncio.Queue()
    hub._queue = queue
    return hub, bus, credit_mgr, queue


@pytest.mark.asyncio
async def test_hub_publishes_and_consumes_credit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hub now consumes credit before publishing and emits no control event when credits remain."""

    hub, bus, credit_mgr, queue = _configure_hub(monkeypatch)

    payload = {"agent_id": "gap-agent", "timestamp": "2025-10-24T12:05:00Z"}
    await queue.put(("gap-run", "step", payload))

    drain_task = asyncio.create_task(hub._drain_loop())
    await _wait_until(lambda: bool(bus.published))
    hub._stopping = True
    drain_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await drain_task

    assert bus.published, "TelemetryAsyncHub should publish at least one event"
    assert credit_mgr.initialize_calls == [("gap-run", "gap-agent")]
    assert credit_mgr.consume_calls == [("gap-run", "gap-agent")]
    assert all(event.topic is streams.Topic.STEP_APPENDED for event in bus.published)
    assert not any(event.topic is streams.Topic.CONTROL for event in bus.published)


@pytest.mark.asyncio
async def test_hub_burst_emits_control_on_starvation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Burst traffic that exhausts credits emits CONTROL events and still publishes steps."""

    hub, bus, credit_mgr, queue = _configure_hub(monkeypatch)
    credit_mgr.consume_return = False

    drain_task = asyncio.create_task(hub._drain_loop())

    payload = {"agent_id": "burst-agent", "timestamp": "2025-10-24T12:06:00Z"}
    for step_index in range(24):
        await queue.put(("burst-run", "step", dict(payload, step_index=step_index)))

    await _wait_until(lambda: len(bus.published) >= 24, timeout=2.0)
    hub._stopping = True
    drain_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await drain_task

    control_events = [evt for evt in bus.published if evt.topic is streams.Topic.CONTROL]
    step_events = [evt for evt in bus.published if evt.topic is streams.Topic.STEP_APPENDED]

    assert len(step_events) == 24
    assert len(control_events) == 1, "Expected a single CONTROL event for starvation"
    assert control_events[0].payload.get("state") == "STARVED"
    assert credit_mgr.consume_calls == [("burst-run", "burst-agent")] * 24
