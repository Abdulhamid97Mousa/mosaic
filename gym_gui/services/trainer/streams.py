import asyncio
import errno
from collections import deque
from dataclasses import dataclass
import logging
import threading
from typing import Any, Deque, Dict, Optional

from qtpy import QtCore


_LOGGER = logging.getLogger("gym_gui.trainer.streams")


def _proto_to_dict(proto_msg: Any) -> dict[str, Any]:
    """Convert protobuf message to dictionary for UI consumption."""
    from google.protobuf.json_format import MessageToDict
    return MessageToDict(
        proto_msg, 
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True  # Include empty strings, zeros, etc.
    )


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
    step_received = QtCore.Signal(object)  # type: ignore[attr-defined]
    episode_received = QtCore.Signal(object)  # type: ignore[attr-defined]
    queue_overflow = QtCore.Signal(str, str, int)  # type: ignore[attr-defined]
    run_completed = QtCore.Signal(str)  # type: ignore[attr-defined]  # NEW: emits run_id when run finishes

    def emit_step(self, message: TelemetryStep) -> None:
        # Post custom event for thread-safe delivery to main thread
        QtCore.QCoreApplication.postEvent(self, _TelemetryEvent("step", message))

    def emit_episode(self, message: TelemetryEpisode) -> None:
        # Post custom event for thread-safe delivery to main thread
        QtCore.QCoreApplication.postEvent(self, _TelemetryEvent("episode", message))

    def emit_overflow(self, run_id: str, stream_type: str, dropped: int) -> None:
        # Post custom event for thread-safe delivery to main thread
        QtCore.QCoreApplication.postEvent(self, _TelemetryEvent("overflow", (run_id, stream_type, dropped)))

    def emit_run_completed(self, run_id: str) -> None:
        """Signal that a training run has completed."""
        # Post custom event for thread-safe delivery to main thread
        QtCore.QCoreApplication.postEvent(self, _TelemetryEvent("completed", run_id))

    def event(self, e: QtCore.QEvent) -> bool:
        """Handle custom telemetry events on the main thread."""
        if isinstance(e, _TelemetryEvent):
            if e.event_name == "step":
                self.step_received.emit(e.data)
            elif e.event_name == "episode":
                self.episode_received.emit(e.data)
            elif e.event_name == "overflow":
                run_id, stream_type, dropped = e.data  # type: ignore[misc]
                self.queue_overflow.emit(run_id, stream_type, dropped)
            elif e.event_name == "completed":
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
            self.dropped_steps += 1
            return self.dropped_steps
        return None

    def add_episode(self, episode: TelemetryEpisode) -> Optional[int]:
        before = len(self.episodes)
        self.episodes.append(episode)
        if len(self.episodes) == before:
            self.dropped_episodes += 1
            return self.dropped_episodes
        return None


class TelemetryAsyncHub:
    def __init__(self, *, max_queue: int = 1024, buffer_size: int = 256) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop.set_exception_handler(self._loop_exception_handler)
        self._queue: asyncio.Queue[tuple[str, str, Any]] = asyncio.Queue(maxsize=max_queue)
        self._buffers: Dict[str, RunStreamBuffer] = {}
        self._buffer_size = buffer_size
        self.bridge = TelemetryBridge()
        self._task: Optional[asyncio.Task[None]] = None
        self._thread: Optional[threading.Thread] = None
        self._subscriptions: Dict[str, Dict[str, asyncio.Task[None]]] = {}  # run_id -> {stream_type: task}
        self._stopping = False
        self._started = False
        self._completed: set[str] = set()  # Guard against multiple run_completed signals

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopping = False
        _LOGGER.info("Telemetry hub event loop starting", extra={"thread_id": threading.get_ident()})
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="telemetry-hub-loop",
            daemon=True,
        )
        self._thread.start()
        # Schedule drain loop inside the running loop
        self._call_soon_threadsafe(self._start_drain_loop)
        self._started = True
        _LOGGER.info("Telemetry hub thread started successfully")

    def _start_drain_loop(self) -> None:
        if self._task is None:
            self._task = self._loop.create_task(self._drain_loop())

    def _call_soon_threadsafe(self, callback, *args) -> None:
        self._loop.call_soon_threadsafe(callback, *args)

    def is_running(self) -> bool:
        return self._started and self._thread is not None

    def _loop_exception_handler(self, loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, BlockingIOError) and getattr(exc, "errno", None) in {errno.EAGAIN, errno.EWOULDBLOCK}:
            _LOGGER.debug(
                "Ignoring non-fatal BlockingIOError from gRPC poller",
                extra={"grpc_message": context.get("message")},  # Rename to avoid conflict
            )
            return

        # Sanitize context to avoid LogRecord key conflicts
        sanitized_context = {}
        for key, value in context.items():
            if key == "message":
                # Rename conflicting 'message' key to avoid LogRecord conflict
                sanitized_context["exception_message"] = value
            else:
                sanitized_context[key] = value

        _LOGGER.error(
            "Unhandled exception in telemetry hub loop",
            extra=sanitized_context,
        )

    def subscribe_run(self, run_id: str, client: Any) -> None:
        """Subscribe to both step and episode streams for a run."""
        if not self.is_running():
            self.start()
        _LOGGER.info("Subscribing to telemetry for run", extra={"run_id": run_id})
        self._call_soon_threadsafe(self._subscribe_run_async, run_id, client)

    def _subscribe_run_async(self, run_id: str, client: Any) -> None:
        if run_id in self._subscriptions:
            _LOGGER.debug("Already subscribed to run", extra={"run_id": run_id})
            return
        _LOGGER.info("Creating telemetry stream tasks for run: %s", run_id)
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
        _LOGGER.info("Subscribed to telemetry streams (tasks created): %s", run_id)

    async def _stream_steps(self, run_id: str, client: Any) -> None:
        last_seq = 0
        _LOGGER.debug("Starting step stream for run: %s", run_id)
        try:
            async with client.stream_run_steps(run_id, since_seq=0) as stream:
                _LOGGER.debug("Step stream opened for run: %s, waiting for data...", run_id)
                async for payload in stream:
                    if self._stopping:
                        break
                    seq_id = getattr(payload, "seq_id", -1)
                    _LOGGER.debug("Received step: run=%s seq=%d ep=%d step=%d", 
                                  run_id, seq_id, 
                                  getattr(payload, "episode_index", -1),
                                  getattr(payload, "step_index", -1))
                    if last_seq > 0:
                        gap = seq_id - last_seq - 1
                        if gap > 0:
                            _LOGGER.warning(
                                "Detected sequence gap in steps",
                                extra={"run_id": run_id, "gap": gap, "last_seq": last_seq, "seq_id": seq_id},
                            )
                    last_seq = seq_id
                    try:
                        self._queue.put_nowait((run_id, "step", payload))
                    except asyncio.QueueFull:
                        self.bridge.emit_overflow(run_id, "step", 1)
        except asyncio.CancelledError:
            _LOGGER.debug("Step stream cancelled", extra={"run_id": run_id})
        except Exception as exc:
            _LOGGER.exception("Step stream error", exc_info=exc, extra={"run_id": run_id})
        finally:
            _LOGGER.info("Step stream closed", extra={"run_id": run_id})
            # Emit run_completed signal only once
            if run_id not in self._completed:
                self._completed.add(run_id)
                self.bridge.emit_run_completed(run_id)
                # Schedule unsubscribe to clean up
                self._loop.call_soon(self._unsubscribe_run_async, run_id)

    async def _stream_episodes(self, run_id: str, client: Any) -> None:
        last_seq = 0
        try:
            async with client.stream_run_episodes(run_id, since_seq=0) as stream:
                async for payload in stream:
                    if self._stopping:
                        break
                    seq_id = getattr(payload, "seq_id", -1)
                    if last_seq > 0:
                        gap = seq_id - last_seq - 1
                        if gap > 0:
                            _LOGGER.warning(
                                "Detected sequence gap in episodes",
                                extra={"run_id": run_id, "gap": gap, "last_seq": last_seq, "seq_id": seq_id},
                            )
                    last_seq = seq_id
                    try:
                        self._queue.put_nowait((run_id, "episode", payload))
                    except asyncio.QueueFull:
                        self.bridge.emit_overflow(run_id, "episode", 1)
        except asyncio.CancelledError:
            _LOGGER.debug("Episode stream cancelled", extra={"run_id": run_id})
        except Exception as exc:
            _LOGGER.exception("Episode stream error", exc_info=exc, extra={"run_id": run_id})
        finally:
            _LOGGER.info("Episode stream closed", extra={"run_id": run_id})

    def unsubscribe_run(self, run_id: str) -> None:
        """Cancel subscriptions and clear buffers for a run."""
        self._call_soon_threadsafe(self._unsubscribe_run_async, run_id)

    def _unsubscribe_run_async(self, run_id: str) -> None:
        _LOGGER.info("Unsubscribing from telemetry for run", extra={"run_id": run_id})
        tasks = self._subscriptions.pop(run_id, None)
        if tasks:
            for task in tasks.values():
                task.cancel()
        self._buffers.pop(run_id, None)
        _LOGGER.debug("Unsubscribed and cleaned up buffers for run: %s", run_id)
        _LOGGER.info("Unsubscribed from telemetry streams", extra={"run_id": run_id})

    async def _drain_loop(self) -> None:
        while not self._stopping:
            try:
                run_id, stream_type, payload = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            buffer = self._buffers.setdefault(run_id, RunStreamBuffer(self._buffer_size))
            if stream_type == "step":
                # Convert protobuf to dict for UI consumption
                payload_dict = _proto_to_dict(payload)
                step = TelemetryStep(run_id, payload_dict, getattr(payload, "seq_id", -1))
                overflow = buffer.add_step(step)
                if overflow is not None:
                    self.bridge.emit_overflow(run_id, stream_type, overflow)
                else:
                    self.bridge.emit_step(step)
            else:
                # Convert protobuf to dict for UI consumption
                payload_dict = _proto_to_dict(payload)
                episode = TelemetryEpisode(run_id, payload_dict, getattr(payload, "seq_id", -1))
                overflow = buffer.add_episode(episode)
                if overflow is not None:
                    self.bridge.emit_overflow(run_id, stream_type, overflow)
                else:
                    self.bridge.emit_episode(episode)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stopping = True
        # Cancel all subscriptions
        for run_id in list(self._subscriptions.keys()):
            self._call_soon_threadsafe(self._unsubscribe_run_async, run_id)
        # Cancel drain task
        if self._task is not None:
            self._call_soon_threadsafe(self._task.cancel)
        # Stop loop
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
        self._thread = None
        self._task = None
        self._completed.clear()
        _LOGGER.info("Telemetry hub stopped")

    def submit_step(self, run_id: str, payload: Any) -> None:
        self._submit(run_id, "step", payload)

    def submit_episode(self, run_id: str, payload: Any) -> None:
        self._submit(run_id, "episode", payload)

    def _submit(self, run_id: str, stream_type: str, payload: Any) -> None:
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
