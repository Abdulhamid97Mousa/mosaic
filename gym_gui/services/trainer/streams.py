# /home/hamid/Desktop/Projects/GUI_BDI_RL/gym_gui/services/trainer/streams.py

import asyncio
import errno
from collections import deque
from dataclasses import dataclass
import logging
import threading
from typing import Any, Deque, Dict, Optional, TYPE_CHECKING

from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]

from gym_gui.telemetry.events import Topic, TelemetryEvent
from gym_gui.telemetry.run_bus import get_bus
from gym_gui.telemetry.credit_manager import get_credit_manager
from gym_gui.constants import (
    TELEMETRY_HUB_MAX_QUEUE,
    TELEMETRY_HUB_BUFFER_SIZE,
)
from gym_gui.logging_config.log_constants import (
    LOG_SERVICE_TELEMETRY_BRIDGE_STEP_QUEUED,
    LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_QUEUED,
    LOG_SERVICE_TELEMETRY_BRIDGE_STEP_DELIVERED,
    LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_DELIVERED,
    LOG_SERVICE_TELEMETRY_BRIDGE_OVERFLOW,
    LOG_SERVICE_TELEMETRY_BRIDGE_RUN_COMPLETED,
    LOG_SERVICE_TELEMETRY_HUB_STARTED,
    LOG_SERVICE_TELEMETRY_HUB_SUBSCRIBED,
    LOG_SERVICE_TELEMETRY_HUB_TRACE,
    LOG_SERVICE_TELEMETRY_HUB_ERROR,
)
from gym_gui.logging_config.helpers import log_constant

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gym_gui.telemetry.run_bus import RunBus

_LOGGER = logging.getLogger("gym_gui.trainer.streams")


def _proto_to_dict(proto_msg: Any) -> dict[str, Any]:
    """Convert protobuf message to dictionary for UI consumption."""
    from google.protobuf.json_format import MessageToDict
    return MessageToDict(
        proto_msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True  # Include empty strings, zeros, etc.
    )


def _normalize_payload(payload: Any, run_id: str, default_agent_id: str = "default") -> dict[str, Any]:
    """
    Normalize telemetry payload to consistent dictionary format.

    Ensures all downstream consumers receive a consistent dictionary with required fields:
    - run_id (string)
    - agent_id (string)
    - episode_index (integer)
    - step_index (integer)
    - payload_version (integer, for future compatibility)
    - All other telemetry data fields

    Args:
        payload: Either a dict or protobuf object
        run_id: The run ID for this telemetry
        default_agent_id: Default agent ID if not found in payload

    Returns:
        Normalized dictionary with all required fields
    """
    # Convert protobuf to dict if needed
    if not isinstance(payload, dict):
        payload = _proto_to_dict(payload)

    # Create normalized payload with required fields
    normalized: dict[str, Any] = dict(payload)  # Copy all existing fields

    # Ensure required fields exist
    normalized.setdefault("run_id", run_id)
    normalized.setdefault("agent_id", payload.get("agent_id") or default_agent_id)

    # Normalize episode index (handle both "episode" and "episode_index")
    if "episode_index" not in normalized:
        normalized["episode_index"] = normalized.get("episode", 0)

    # Normalize step index (handle both "step" and "step_index")
    if "step_index" not in normalized:
        normalized["step_index"] = normalized.get("step", 0)

    # Add payload version for future compatibility
    normalized.setdefault("payload_version", 1)

    return normalized


@dataclass(slots=True)
class TelemetryStep:
    run_id: str
    payload: Any
    seq_id: int


@dataclass(slots=True)
class TelemetryEpisode:
    run_id: str
    payload: Any
    seq_id: int


class _TelemetryEvent(QtCore.QEvent):
    """Custom event for thread-safe telemetry delivery."""
    EVENT_TYPE = QtCore.QEvent.Type(QtCore.QEvent.registerEventType())
    
    def __init__(self, event_name: str, data: object) -> None:
        super().__init__(self.EVENT_TYPE)
        self.event_name = event_name
        self.data = data


class TelemetryBridge(QtCore.QObject):
    step_received = pyqtSignal(object)
    episode_received = pyqtSignal(object)
    queue_overflow = pyqtSignal(str, str, int)
    run_completed = pyqtSignal(str)  # NEW: emits run_id when run finishes

    def emit_step(self, message: TelemetryStep) -> None:
        # Normalize payload to ensure consistent dictionary format
        normalized_payload = _normalize_payload(message.payload, message.run_id)
        normalized_message = TelemetryStep(message.run_id, normalized_payload, message.seq_id)

        # Post custom event for thread-safe delivery to main thread
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_BRIDGE_STEP_QUEUED, extra={"run_id": message.run_id, "seq": message.seq_id})
        QtCore.QCoreApplication.postEvent(self, _TelemetryEvent("step", normalized_message))

    def emit_episode(self, message: TelemetryEpisode) -> None:
        # Normalize payload to ensure consistent dictionary format
        normalized_payload = _normalize_payload(message.payload, message.run_id)
        normalized_message = TelemetryEpisode(message.run_id, normalized_payload, message.seq_id)

        # Post custom event for thread-safe delivery to main thread
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_QUEUED, extra={"run_id": message.run_id, "seq": message.seq_id})
        QtCore.QCoreApplication.postEvent(self, _TelemetryEvent("episode", normalized_message))

    def emit_overflow(self, run_id: str, stream_type: str, dropped: int) -> None:
        # Post custom event for thread-safe delivery to main thread
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_BRIDGE_OVERFLOW, extra={"run_id": run_id, "stream_type": stream_type, "dropped": dropped})
        QtCore.QCoreApplication.postEvent(self, _TelemetryEvent("overflow", (run_id, stream_type, dropped)))

    def emit_run_completed(self, run_id: str) -> None:
        """Signal that a training run has completed."""
        # Post custom event for thread-safe delivery to main thread
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_BRIDGE_RUN_COMPLETED, extra={"run_id": run_id})
        QtCore.QCoreApplication.postEvent(self, _TelemetryEvent("completed", run_id))

    def event(self, e: QtCore.QEvent) -> bool:
        """Handle custom telemetry events on the main thread."""
        if isinstance(e, _TelemetryEvent):
            if e.event_name == "step":
                payload = getattr(e, "data", None)
                log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_BRIDGE_STEP_DELIVERED, extra={"run_id": getattr(payload, "run_id", None), "seq": getattr(payload, "seq_id", None)})
                self.step_received.emit(e.data)
            elif e.event_name == "episode":
                payload = getattr(e, "data", None)
                log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_DELIVERED, extra={"run_id": getattr(payload, "run_id", None), "seq": getattr(payload, "seq_id", None)})
                self.episode_received.emit(e.data)
            elif e.event_name == "overflow":
                run_id, stream_type, dropped = e.data  # type: ignore[misc]
                log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_BRIDGE_OVERFLOW, extra={"run_id": run_id, "stream_type": stream_type, "dropped": dropped})
                self.queue_overflow.emit(run_id, stream_type, dropped)
            elif e.event_name == "completed":
                log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_BRIDGE_RUN_COMPLETED, extra={"run_id": e.data})
                self.run_completed.emit(e.data)  # type: ignore[arg-type]
            return True
        return super().event(e)


class RunStreamBuffer:
    def __init__(self, maxlen: int) -> None:
        self.steps: Deque[TelemetryStep] = deque(maxlen=maxlen)
        self.episodes: Deque[TelemetryEpisode] = deque(maxlen=maxlen)
        self.dropped_steps = 0
        self.dropped_episodes = 0

    def add_step(self, step: TelemetryStep) -> Optional[int]:
        before = len(self.steps)
        self.steps.append(step)
        if len(self.steps) == before:
            # Buffer was full, deque dropped the oldest item
            self.dropped_steps += 1
            return 1  # Return 1 to indicate a single drop occurred (not cumulative)
        return None

    def add_episode(self, episode: TelemetryEpisode) -> Optional[int]:
        before = len(self.episodes)
        self.episodes.append(episode)
        if len(self.episodes) == before:
            # Buffer was full, deque dropped the oldest item
            self.dropped_episodes += 1
            return 1  # Return 1 to indicate a single drop occurred (not cumulative)
        return None


class TelemetryAsyncHub:
    def __init__(
        self,
        *,
        max_queue: int = TELEMETRY_HUB_MAX_QUEUE,
        buffer_size: int = TELEMETRY_HUB_BUFFER_SIZE,
    ) -> None:
        # Defer loop detection until start() is called
        # This allows the Qt event loop to be set up first
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._owns_loop = False
        self._max_queue = max_queue
        self._buffer_size = buffer_size  # Default buffer size
        self._run_buffer_sizes: Dict[str, int] = {}  # Per-run buffer size overrides
        self._queue: Optional[asyncio.Queue[tuple[str, str, Any]]] = None
        self._buffers: Dict[str, RunStreamBuffer] = {}
        self.bridge = TelemetryBridge()
        self._task: Optional[asyncio.Task[None]] = None
        self._thread: Optional[threading.Thread] = None
        self._subscriptions: Dict[str, Dict[str, asyncio.Task[None]]] = {}  # run_id -> {stream_type: task}
        self._stopping = False
        self._started = False
        self._completed: set[str] = set()  # Guard against multiple run_completed signals
        self._credit_mgr = get_credit_manager()  # Shared credit manager for UI backpressure
        self._starved_streams: set[tuple[str, str, str]] = set()
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"max_queue": max_queue, "buffer_size": buffer_size})

    def start(self) -> None:
        if self._started:
            return
        self._stopping = False
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_STARTED)

        # Initialize loop on first start (deferred from __init__)
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
                self._owns_loop = False
                log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE)
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                self._owns_loop = True
                log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE)

            self._loop.set_exception_handler(self._loop_exception_handler)
            self._queue = asyncio.Queue(maxsize=self._max_queue)

        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"owns_loop": self._owns_loop, "loop": str(self._loop)})

        # If we own the loop, run it in a thread
        if self._owns_loop:
            self._thread = threading.Thread(
                target=self._loop.run_forever,
                name="telemetry-hub-loop",
                daemon=True,
            )
            self._thread.start()
            self._call_soon_threadsafe(self._start_drain_loop)
        else:
            # Using Qt event loop - schedule drain loop directly
            self._start_drain_loop()

        self._started = True
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_STARTED)

    def _start_drain_loop(self) -> None:
        if self._task is None and self._loop is not None:
            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE)
            self._task = self._loop.create_task(self._drain_loop())

    def _call_soon_threadsafe(self, callback, *args) -> None:
        if self._loop is None:
            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_ERROR)
            return
        if self._owns_loop and self._thread is not None:
            self._loop.call_soon_threadsafe(lambda: callback(*args))
        else:
            # Already in the right loop - call directly
            callback(*args)

    def is_running(self) -> bool:
        return self._started

    def _loop_exception_handler(self, _loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, BlockingIOError) and getattr(exc, "errno", None) in {errno.EAGAIN, errno.EWOULDBLOCK}:
            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"grpc_message": context.get("message")})
            return

        # Sanitize context to avoid LogRecord key conflicts
        sanitized_context = {}
        for key, value in context.items():
            if key == "message":
                # Rename conflicting 'message' key to avoid LogRecord conflict
                sanitized_context["exception_message"] = value
            else:
                sanitized_context[key] = value

        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_ERROR, extra=sanitized_context)

    def set_run_buffer_size(self, run_id: str, buffer_size: int) -> None:
        """Set custom buffer size for a specific run.
        
        This should be called BEFORE subscribing to the run to ensure the buffer
        is created with the correct size based on training configuration.
        
        Args:
            run_id: The run identifier
            buffer_size: Buffer size for this run's telemetry stream
        """
        self._run_buffer_sizes[run_id] = buffer_size
        log_constant(
            _LOGGER,
            LOG_SERVICE_TELEMETRY_HUB_TRACE,
            extra={"run_id": run_id, "buffer_size": buffer_size, "action": "set_run_buffer_size"}
        )

    def subscribe_run(self, run_id: str, client: Any) -> None:
        """Subscribe to both step and episode streams for a run."""
        if not self.is_running():
            self.start()

        if self._loop is None:
            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_ERROR, extra={"run_id": run_id})
            return

        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_SUBSCRIBED, extra={"run_id": run_id})

        # Schedule subscription in the event loop
        if self._owns_loop:
            self._call_soon_threadsafe(self._subscribe_run_async, run_id, client)
        else:
            # Already in the Qt event loop, schedule as a task
            asyncio.create_task(self._subscribe_run_async_wrapper(run_id, client))

        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id, "owns_loop": self._owns_loop})

    async def _subscribe_run_async_wrapper(self, run_id: str, client: Any) -> None:
        """Wrapper to call _subscribe_run_async from async context."""
        self._subscribe_run_async(run_id, client)

    def _subscribe_run_async(self, run_id: str, client: Any) -> None:
        if run_id in self._subscriptions:
            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id})
            return
        if self._loop is None:
            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_ERROR, extra={"run_id": run_id})
            return
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_SUBSCRIBED, extra={"run_id": run_id})
        tasks = {}
        tasks["step"] = self._loop.create_task(
            self._stream_steps(run_id, client),
            name=f"stream-steps-{run_id}",
        )
        tasks["episode"] = self._loop.create_task(
            self._stream_episodes(run_id, client),
            name=f"stream-episodes-{run_id}",
        )
        self._subscriptions[run_id] = tasks
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_SUBSCRIBED, extra={"run_id": run_id})
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id})

    async def _stream_steps(self, run_id: str, client: Any) -> None:
        if self._queue is None:
            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_ERROR, extra={"run_id": run_id})
            return

        last_seq = 0
        reconnect_attempts = 0
        max_reconnect_attempts = 10
        reconnect_delay = 1.0  # Start with 1 second delay

        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id})
        try:
            while not self._stopping:
                try:
                    async with client.stream_run_steps(run_id, since_seq=last_seq) as stream:
                        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id})
                        reconnect_attempts = 0  # Reset on successful connection
                        reconnect_delay = 1.0

                        async for payload in stream:
                            if self._stopping:
                                break
                            seq_id = getattr(payload, "seq_id", -1)
                            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={
                                "run_id": run_id, "seq": seq_id,
                                "episode_index": getattr(payload, "episode_index", -1),
                                "step_index": getattr(payload, "step_index", -1)
                            })
                            if last_seq > 0:
                                gap = seq_id - last_seq - 1
                                if gap > 0:
                                    log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id, "gap": gap, "last_seq": last_seq, "seq_id": seq_id})
                            last_seq = seq_id
                            try:
                                self._queue.put_nowait((run_id, "step", payload))
                                log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={
                                    "run_id": run_id,
                                    "seq_id": seq_id,
                                    "qsize": self._queue.qsize() if hasattr(self._queue, "qsize") else None,
                                })
                            except asyncio.QueueFull:
                                self.bridge.emit_overflow(run_id, "step", 1)

                        # Stream closed normally, attempt reconnection
                        if not self._stopping:
                            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id, "last_seq": last_seq})
                            await asyncio.sleep(0.5)  # Brief delay before reconnect
                            continue
                        else:
                            break

                except asyncio.CancelledError:
                    log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id})
                    break
                except Exception as exc:
                    reconnect_attempts += 1
                    if reconnect_attempts >= max_reconnect_attempts:
                        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_ERROR, extra={"run_id": run_id, "attempts": reconnect_attempts, "error": str(exc)})
                        break
                    log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id, "attempt": reconnect_attempts, "error": str(exc)})
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, 30.0)  # Exponential backoff, max 30s

        except asyncio.CancelledError:
            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id})
        finally:
            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id})
            # NOTE: Do NOT emit run_completed here. Stream closure does not mean training is complete.
            # The stream may close due to:
            # - Episode boundary (normal, training continues)
            # - Network hiccup (stream will reconnect)
            # - Worker reconnection (new stream created)
            # Only the worker should signal training completion via explicit message.
            # Unsubscribe only if explicitly requested by caller.

    async def _stream_episodes(self, run_id: str, client: Any) -> None:
        if self._queue is None:
            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_ERROR, extra={"run_id": run_id})
            return

        last_seq = 0
        reconnect_attempts = 0
        max_reconnect_attempts = 10
        reconnect_delay = 1.0  # Start with 1 second delay

        try:
            while not self._stopping:
                try:
                    async with client.stream_run_episodes(run_id, since_seq=last_seq) as stream:
                        reconnect_attempts = 0  # Reset on successful connection
                        reconnect_delay = 1.0

                        async for payload in stream:
                            if self._stopping:
                                break
                            seq_id = getattr(payload, "seq_id", -1)
                            if last_seq > 0:
                                gap = seq_id - last_seq - 1
                                if gap > 0:
                                    log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id, "gap": gap, "last_seq": last_seq, "seq_id": seq_id})
                            last_seq = seq_id
                            try:
                                self._queue.put_nowait((run_id, "episode", payload))
                                log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={
                                    "run_id": run_id,
                                    "seq_id": seq_id,
                                    "qsize": self._queue.qsize() if hasattr(self._queue, "qsize") else None,
                                })
                            except asyncio.QueueFull:
                                self.bridge.emit_overflow(run_id, "episode", 1)

                        # Stream closed normally, attempt reconnection
                        if not self._stopping:
                            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id, "last_seq": last_seq})
                            await asyncio.sleep(0.5)  # Brief delay before reconnect
                            continue
                        else:
                            break

                except asyncio.CancelledError:
                    log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id})
                    break
                except Exception as exc:
                    reconnect_attempts += 1
                    if reconnect_attempts >= max_reconnect_attempts:
                        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_ERROR, extra={"run_id": run_id, "attempts": reconnect_attempts, "error": str(exc)})
                        break
                    log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id, "attempt": reconnect_attempts, "error": str(exc)})
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, 30.0)  # Exponential backoff, max 30s

        except asyncio.CancelledError:
            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id})
        finally:
            log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id})

    def unsubscribe_run(self, run_id: str) -> None:
        """Cancel subscriptions and clear buffers for a run."""
        self._call_soon_threadsafe(self._unsubscribe_run_async, run_id)

    def _unsubscribe_run_async(self, run_id: str) -> None:
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_SUBSCRIBED, extra={"run_id": run_id})
        tasks = self._subscriptions.pop(run_id, None)
        if tasks:
            for name, task in tasks.items():
                log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id, "task": name})
                task.cancel()
        self._buffers.pop(run_id, None)
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_TRACE, extra={"run_id": run_id})
        log_constant(_LOGGER, LOG_SERVICE_TELEMETRY_HUB_SUBSCRIBED, extra={"run_id": run_id})

    def _publish_control_event(
        self,
        bus: "RunBus",
        *,
        run_id: str,
        agent_id: str,
        seq_id: int,
        timestamp: str,
        stream_type: str,
        state: str,
    ) -> None:
        """Publish a control-plane event for credit state changes."""

        control_evt = TelemetryEvent(
            topic=Topic.CONTROL,
            run_id=run_id,
            agent_id=agent_id,
            seq_id=seq_id,
            ts_iso=timestamp,
            payload={
                "state": state,
                "stream_type": stream_type,
                "agent_id": agent_id,
            },
        )
        bus.publish(control_evt)

    async def _drain_loop(self) -> None:
        if self._queue is None:
            _LOGGER.error("Queue not initialized in _drain_loop")
            return

        _LOGGER.debug("Telemetry drain loop running")
        while not self._stopping:
            try:
                run_id, stream_type, payload = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
                _LOGGER.debug(
                    "Drain loop: dequeued payload",
                    extra={
                        "run_id": run_id,
                        "stream_type": stream_type,
                        "qsize": self._queue.qsize() if hasattr(self._queue, "qsize") else None,
                    },
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Use per-run buffer size if configured, otherwise use default
            run_buffer_size = self._run_buffer_sizes.get(run_id, self._buffer_size)
            buffer = self._buffers.setdefault(run_id, RunStreamBuffer(run_buffer_size))
            if stream_type == "step":
                # Convert protobuf to dict for UI consumption
                payload_dict = _proto_to_dict(payload)
                step = TelemetryStep(run_id, payload_dict, getattr(payload, "seq_id", -1))
                overflow = buffer.add_step(step)
                if overflow is not None:
                    _LOGGER.warning("Step buffer overflow", extra={"run_id": run_id, "overflow": overflow})
                    self.bridge.emit_overflow(run_id, stream_type, overflow)
                else:
                    agent_id = payload_dict.get("agent_id", "default")
                    timestamp = payload_dict.get("timestamp", "")
                    stream_key = (run_id, agent_id, stream_type)

                    try:
                        bus = get_bus()
                        self._credit_mgr.initialize_stream(run_id, agent_id)
                        credits_available = self._credit_mgr.consume_credit(run_id, agent_id)

                        if credits_available:
                            if stream_key in self._starved_streams:
                                self._starved_streams.discard(stream_key)
                                self._publish_control_event(
                                    bus,
                                    run_id=run_id,
                                    agent_id=agent_id,
                                    seq_id=step.seq_id,
                                    timestamp=timestamp,
                                    stream_type=stream_type,
                                    state="RESUMED",
                                )
                                _LOGGER.info(
                                    "Credits replenished; CONTROL RESUMED emitted",
                                    extra={"run_id": run_id, "agent_id": agent_id, "stream_type": stream_type},
                                )

                            _LOGGER.debug(
                                "Emitting bridge.step_received",
                                extra={"run_id": run_id, "seq": step.seq_id},
                            )
                            self.bridge.emit_step(step)
                        else:
                            if stream_key not in self._starved_streams:
                                self._starved_streams.add(stream_key)
                                self._publish_control_event(
                                    bus,
                                    run_id=run_id,
                                    agent_id=agent_id,
                                    seq_id=step.seq_id,
                                    timestamp=timestamp,
                                    stream_type=stream_type,
                                    state="STARVED",
                                )
                                _LOGGER.warning(
                                    "Credits exhausted; CONTROL STARVED emitted",
                                    extra={"run_id": run_id, "agent_id": agent_id, "stream_type": stream_type},
                                )

                            _LOGGER.debug(
                                "Skipping bridge emit due to credit starvation",
                                extra={"run_id": run_id, "agent_id": agent_id},
                            )

                        evt = TelemetryEvent(
                            topic=Topic.STEP_APPENDED,
                            run_id=run_id,
                            agent_id=agent_id,
                            seq_id=step.seq_id,
                            ts_iso=timestamp,
                            payload=payload_dict,
                        )
                        bus.publish(evt)
                        _LOGGER.debug(
                            "Published STEP_APPENDED to RunBus",
                            extra={
                                "run_id": run_id,
                                "agent_id": agent_id,
                                "seq_id": step.seq_id,
                                "event_type": "STEP_APPENDED",
                            },
                        )
                    except Exception as e:
                        _LOGGER.warning(
                            "Failed to publish STEP_APPENDED to RunBus",
                            extra={"run_id": run_id, "error": str(e)},
                        )
            else:
                # Convert protobuf to dict for UI consumption
                payload_dict = _proto_to_dict(payload)
                episode = TelemetryEpisode(run_id, payload_dict, getattr(payload, "seq_id", -1))
                overflow = buffer.add_episode(episode)
                if overflow is not None:
                    _LOGGER.warning("Episode buffer overflow", extra={"run_id": run_id, "overflow": overflow})
                    self.bridge.emit_overflow(run_id, stream_type, overflow)
                else:
                    agent_id = payload_dict.get("agent_id", "default")
                    timestamp = payload_dict.get("timestamp", "")
                    stream_key = (run_id, agent_id, stream_type)

                    try:
                        bus = get_bus()
                        self._credit_mgr.initialize_stream(run_id, agent_id)
                        credits_available = self._credit_mgr.consume_credit(run_id, agent_id)

                        if credits_available:
                            if stream_key in self._starved_streams:
                                self._starved_streams.discard(stream_key)
                                self._publish_control_event(
                                    bus,
                                    run_id=run_id,
                                    agent_id=agent_id,
                                    seq_id=episode.seq_id,
                                    timestamp=timestamp,
                                    stream_type=stream_type,
                                    state="RESUMED",
                                )
                                _LOGGER.info(
                                    "Credits replenished; CONTROL RESUMED emitted",
                                    extra={"run_id": run_id, "agent_id": agent_id, "stream_type": stream_type},
                                )

                            _LOGGER.debug(
                                "Emitting bridge.episode_received",
                                extra={"run_id": run_id, "seq": episode.seq_id},
                            )
                            self.bridge.emit_episode(episode)
                        else:
                            if stream_key not in self._starved_streams:
                                self._starved_streams.add(stream_key)
                                self._publish_control_event(
                                    bus,
                                    run_id=run_id,
                                    agent_id=agent_id,
                                    seq_id=episode.seq_id,
                                    timestamp=timestamp,
                                    stream_type=stream_type,
                                    state="STARVED",
                                )
                                _LOGGER.warning(
                                    "Credits exhausted; CONTROL STARVED emitted",
                                    extra={"run_id": run_id, "agent_id": agent_id, "stream_type": stream_type},
                                )

                            _LOGGER.debug(
                                "Skipping bridge episode emit due to credit starvation",
                                extra={"run_id": run_id, "agent_id": agent_id},
                            )

                        evt = TelemetryEvent(
                            topic=Topic.EPISODE_FINALIZED,
                            run_id=run_id,
                            agent_id=agent_id,
                            seq_id=episode.seq_id,
                            ts_iso=timestamp,
                            payload=payload_dict,
                        )
                        bus.publish(evt)
                        _LOGGER.debug(
                            "Published EPISODE_FINALIZED to RunBus",
                            extra={
                                "run_id": run_id,
                                "agent_id": agent_id,
                                "seq_id": episode.seq_id,
                                "event_type": "EPISODE_FINALIZED",
                            },
                        )
                    except Exception as e:
                        _LOGGER.warning(
                            "Failed to publish EPISODE_FINALIZED to RunBus",
                            extra={"run_id": run_id, "error": str(e)},
                        )

            if self._queue is not None:
                self._queue.task_done()

    def stop(self) -> None:
        if not self._started or self._loop is None:
            return
        self._stopping = True
        _LOGGER.info("Stopping telemetry hub")

        # Cancel all subscriptions
        for run_id in list(self._subscriptions.keys()):
            self._call_soon_threadsafe(self._unsubscribe_run_async, run_id)

        # Cancel drain task
        if self._task is not None:
            self._call_soon_threadsafe(self._task.cancel)

        # Stop loop only if we own it
        if self._owns_loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=2)
                self._thread = None

        self._task = None
        self._completed.clear()
        self._started = False
        _LOGGER.info("Telemetry hub stopped")

    def submit_step(self, run_id: str, payload: Any) -> None:
        self._submit(run_id, "step", payload)

    def submit_episode(self, run_id: str, payload: Any) -> None:
        self._submit(run_id, "episode", payload)

    def _submit(self, run_id: str, stream_type: str, payload: Any) -> None:
        if self._queue is None:
            _LOGGER.warning("Queue not initialized, dropping telemetry")
            return
        try:
            self._queue.put_nowait((run_id, stream_type, payload))
        except asyncio.QueueFull:
            self.bridge.emit_overflow(run_id, stream_type, 1)


__all__ = [
    "TelemetryAsyncHub",
    "TelemetryBridge",
    "TelemetryEpisode",
    "TelemetryStep",
]
