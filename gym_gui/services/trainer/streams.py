import asyncio
from collections import deque
from dataclasses import dataclass
import logging
import threading
from typing import Any, Deque, Dict, Optional

from qtpy import QtCore


_LOGGER = logging.getLogger("gym_gui.trainer.streams")


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


class TelemetryBridge(QtCore.QObject):
    step_received = QtCore.Signal(object)  # type: ignore[attr-defined]
    episode_received = QtCore.Signal(object)  # type: ignore[attr-defined]
    queue_overflow = QtCore.Signal(str, str, int)  # type: ignore[attr-defined]
    run_completed = QtCore.Signal(str)  # type: ignore[attr-defined]  # NEW: emits run_id when run finishes

    def emit_step(self, message: TelemetryStep) -> None:
        QtCore.QMetaObject.invokeMethod(
            self,
            "_emit_step",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, message),
        )

    def _emit_step(self, message: TelemetryStep) -> None:
        self.step_received.emit(message)

    def emit_episode(self, message: TelemetryEpisode) -> None:
        QtCore.QMetaObject.invokeMethod(
            self,
            "_emit_episode",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, message),
        )

    def _emit_episode(self, message: TelemetryEpisode) -> None:
        self.episode_received.emit(message)

    def emit_overflow(self, run_id: str, stream_type: str, dropped: int) -> None:
        QtCore.QMetaObject.invokeMethod(
            self,
            "_emit_overflow",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, run_id),
            QtCore.Q_ARG(str, stream_type),
            QtCore.Q_ARG(int, dropped),
        )

    def _emit_overflow(self, run_id: str, stream_type: str, dropped: int) -> None:
        self.queue_overflow.emit(run_id, stream_type, dropped)

    def emit_run_completed(self, run_id: str) -> None:
        """Signal that a training run has completed."""
        QtCore.QMetaObject.invokeMethod(
            self,
            "_emit_run_completed",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, run_id),
        )

    def _emit_run_completed(self, run_id: str) -> None:
        self.run_completed.emit(run_id)


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
        self._queue: asyncio.Queue[tuple[str, str, Any]] = asyncio.Queue(maxsize=max_queue)
        self._buffers: Dict[str, RunStreamBuffer] = {}
        self._buffer_size = buffer_size
        self.bridge = TelemetryBridge()
        self._task: Optional[asyncio.Task[None]] = None
        self._thread: Optional[threading.Thread] = None
        self._subscriptions: Dict[str, Dict[str, asyncio.Task[None]]] = {}  # run_id -> {stream_type: task}
        self._stopping = False
        self._started = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopping = False
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="telemetry-hub-loop",
            daemon=True,
        )
        self._thread.start()
        # Schedule drain loop inside the running loop
        self._call_soon_threadsafe(self._start_drain_loop)
        self._started = True

    def _start_drain_loop(self) -> None:
        if self._task is None:
            self._task = self._loop.create_task(self._drain_loop())

    def _call_soon_threadsafe(self, callback, *args) -> None:
        self._loop.call_soon_threadsafe(callback, *args)

    def is_running(self) -> bool:
        return self._started and self._thread is not None

    def subscribe_run(self, run_id: str, client: Any) -> None:
        """Subscribe to both step and episode streams for a run."""
        if not self.is_running():
            self.start()
        self._call_soon_threadsafe(self._subscribe_run_async, run_id, client)

    def _subscribe_run_async(self, run_id: str, client: Any) -> None:
        if run_id in self._subscriptions:
            _LOGGER.debug("Already subscribed to run", extra={"run_id": run_id})
            return
        _LOGGER.debug("Creating telemetry stream tasks for run: %s", run_id)
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
        _LOGGER.debug("Subscribed to telemetry streams (tasks created): %s", run_id)

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

    def unsubscribe_run(self, run_id: str) -> None:
        """Cancel subscriptions and clear buffers for a run."""
        self._call_soon_threadsafe(self._unsubscribe_run_async, run_id)

    def _unsubscribe_run_async(self, run_id: str) -> None:
        tasks = self._subscriptions.pop(run_id, None)
        if tasks:
            for task in tasks.values():
                task.cancel()
        self._buffers.pop(run_id, None)
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
                step = TelemetryStep(run_id, payload, getattr(payload, "seq_id", -1))
                overflow = buffer.add_step(step)
                if overflow is not None:
                    self.bridge.emit_overflow(run_id, stream_type, overflow)
                else:
                    self.bridge.emit_step(step)
            else:
                episode = TelemetryEpisode(run_id, payload, getattr(payload, "seq_id", -1))
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
