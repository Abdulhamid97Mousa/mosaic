from __future__ import annotations

"""Background thread helper to access :class:`TrainerClient` from Qt controllers."""

import asyncio
import errno
import logging
import threading
from queue import Queue, Empty
from typing import Any, Iterable, Optional, Sequence

from gym_gui.services.trainer.client import TrainerClient, TrainerClientConfig
from gym_gui.services.trainer.registry import RunStatus
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_TRAINER_CLIENT_LOOP_NONFATAL,
    LOG_TRAINER_CLIENT_LOOP_ERROR,
    LOG_TRAINER_CLIENT_SHUTDOWN_WARNING,
)


class TrainerClientRunner(LogConstantMixin):
    """Runs :class:`TrainerClient` coroutines on a dedicated asyncio loop."""

    def __init__(self, client: Optional[TrainerClient] = None, *, name: str = "trainer-client-loop") -> None:
        self._client = client or TrainerClient(TrainerClientConfig())
        self._loop = asyncio.new_event_loop()
        self._loop.set_exception_handler(self._loop_exception_handler)
        self._thread = threading.Thread(target=self._loop.run_forever, name=name, daemon=True)
        self._thread.start()
        self._logger = logging.getLogger("gym_gui.trainer.client_runner")

    # ------------------------------------------------------------------
    def submit_run(self, config_json: str, *, run_id: Optional[str] = None, deadline: Optional[float] = None):
        return self._submit(self._client.submit_run(config_json, run_id=run_id, deadline=deadline))

    def cancel_run(self, run_id: str, *, deadline: Optional[float] = None):
        return self._submit(self._client.cancel_run(run_id, deadline=deadline))

    def list_runs(self, statuses: Optional[Sequence[RunStatus]] = None, *, deadline: Optional[float] = None):
        return self._submit(self._client.list_runs(statuses, deadline=deadline))

    def heartbeat(self, run_id: str, *, deadline: Optional[float] = None):
        return self._submit(self._client.heartbeat(run_id, deadline=deadline))

    def get_health(self, *, deadline: Optional[float] = None):
        return self._submit(self._client.get_health(deadline=deadline))

    # ------------------------------------------------------------------
    def watch_runs(
        self,
        statuses: Optional[Iterable[RunStatus]] = None,
        *,
        deadline: Optional[float] = None,
        since_seq: int = 0,
    ) -> "TrainerWatchSubscription":
        queue: Queue[Any] = Queue()
        stop = threading.Event()
        status_seq = list(statuses) if statuses is not None else None

        async def _consume() -> None:
            try:
                async with self._client.watch_runs(status_seq, deadline=deadline, since_seq=since_seq) as stream:
                    async for record in stream:
                        if stop.is_set():
                            break
                        queue.put(record)
            except Exception as exc:  # pragma: no cover - propagated to consumer
                queue.put(exc)
            finally:
                queue.put(_SENTINEL)

        future = asyncio.run_coroutine_threadsafe(_consume(), self._loop)
        return TrainerWatchSubscription(queue, stop, future)

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        try:
            asyncio.run_coroutine_threadsafe(self._client.close(), self._loop).result(timeout=1)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            # Log but don't raise - shutdown should be tolerant of cleanup failures
            self.log_constant(
                LOG_TRAINER_CLIENT_SHUTDOWN_WARNING,
                message="Failed to close trainer client cleanly during shutdown",
                extra={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                exc_info=exc,
            )
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
        self._loop.close()

    def _submit(self, coro: Any):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _loop_exception_handler(self, loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, BlockingIOError) and exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
            self.log_constant(
                LOG_TRAINER_CLIENT_LOOP_NONFATAL,
                message="grpc_blocking_io_ignored",
                extra={
                    "errno": getattr(exc, "errno", None),
                    "grpc_message": context.get("message"),
                },
            )
            return

        # Sanitize context to avoid LogRecord key conflicts
        sanitized_context: dict[str, Any] = {}
        for key, value in context.items():
            if key in {"message", "exc_info", "stack_info", "args"}:
                sanitized_context[f"loop_{key}"] = value
            else:
                sanitized_context[key] = value

        log_message = sanitized_context.pop("loop_message", "Unhandled exception in trainer client loop")
        extra_payload: dict[str, Any] = {}
        for key, value in sanitized_context.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                extra_payload[key] = value
            else:
                extra_payload[key] = repr(value)
        if exc is not None:
            extra_payload.setdefault("exception_type", type(exc).__name__)
        self.log_constant(
            LOG_TRAINER_CLIENT_LOOP_ERROR,
            message=log_message,
            extra=extra_payload,
            exc_info=exc,
        )


class TrainerWatchSubscription:
    """Thread-safe iterator over run records produced by :meth:`TrainerClient.watch_runs`."""

    def __init__(self, queue: Queue[Any], stop_event: threading.Event, future) -> None:
        self._queue = queue
        self._stop = stop_event
        self._future = future

    def get(self, timeout: Optional[float] = None) -> Any:
        try:
            item = self._queue.get(timeout=timeout)
        except Empty:
            raise TimeoutError("Timed out waiting for trainer updates") from None
        if item is _SENTINEL:
            raise TrainerWatchStopped
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self._stop.set()
        self._future.cancel()
        self._queue.put(_SENTINEL)

class TrainerWatchStopped(Exception):
    """Raised when a watch subscription has ended."""


_SENTINEL = object()

__all__ = ["TrainerClientRunner", "TrainerWatchSubscription", "TrainerWatchStopped"]
