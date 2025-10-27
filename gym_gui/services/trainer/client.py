from __future__ import annotations

"""Async client bridge between the Qt GUI and the trainer daemon."""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Sequence, cast
from weakref import WeakKeyDictionary

import grpc

from . import constants as trainer_constants
from .registry import RunStatus
from gym_gui.services.trainer.proto import trainer_pb2 as trainer_pb2_module
from gym_gui.services.trainer.proto import trainer_pb2_grpc
from gym_gui.logging_config.helpers import LogConstantMixin, log_constant
from gym_gui.logging_config.log_constants import (
    LOG_TRAINER_CLIENT_CONNECTING,
    LOG_TRAINER_CLIENT_CONNECTED,
    LOG_TRAINER_CLIENT_CONNECTION_TIMEOUT,
)

trainer_pb2 = cast(Any, trainer_pb2_module)


@dataclass
class _ClientLoopState:
    """Per-event-loop client state (lock, channel, stub)."""
    lock: asyncio.Lock
    channel: Optional[grpc.aio.Channel] = None
    stub: Optional[trainer_pb2_grpc.TrainerServiceStub] = None


CLIENT_DEFAULTS = trainer_constants.TRAINER_DEFAULTS.client


@dataclass(slots=True)
class TrainerClientConfig:
    target: str = CLIENT_DEFAULTS.target
    deadline: float = CLIENT_DEFAULTS.deadline_s
    keepalive_time: float = CLIENT_DEFAULTS.keepalive_time_s
    keepalive_timeout: float = CLIENT_DEFAULTS.keepalive_timeout_s
    connect_timeout: float = CLIENT_DEFAULTS.connect_timeout_s
    max_message_bytes: int = CLIENT_DEFAULTS.max_message_bytes
    http2_min_ping_interval_ms: int = CLIENT_DEFAULTS.http2_min_ping_interval_ms


class TrainerClientConnectionError(RuntimeError):
    """Raised when the client cannot establish a gRPC channel."""


class TrainerClient(LogConstantMixin):
    """High-level async API used by controllers to reach the daemon."""

    def __init__(self, config: Optional[TrainerClientConfig] = None) -> None:
        self._config = config or TrainerClientConfig()
        # Per-event-loop state cache to avoid "Lock bound to different event loop" errors
        self._loop_state: WeakKeyDictionary[asyncio.AbstractEventLoop, _ClientLoopState] = WeakKeyDictionary()
        self._logger = logging.getLogger("gym_gui.trainer.client")

    def _get_or_create_state(self) -> _ClientLoopState:
        """Get or create the state for the current event loop."""
        loop = asyncio.get_running_loop()
        if loop not in self._loop_state:
            self._loop_state[loop] = _ClientLoopState(lock=asyncio.Lock())
        return self._loop_state[loop]

    async def ensure_connected(self) -> trainer_pb2_grpc.TrainerServiceStub:
        state = self._get_or_create_state()
        async with state.lock:
            if state.stub is not None and state.channel is not None:
                return state.stub
            log_constant(self._logger, LOG_TRAINER_CLIENT_CONNECTING, message="Connecting to trainer daemon", extra={"target": self._config.target})
            options = (
                ("grpc.keepalive_time_ms", int(self._config.keepalive_time * 1000)),
                ("grpc.keepalive_timeout_ms", int(self._config.keepalive_timeout * 1000)),
                ("grpc.max_send_message_length", self._config.max_message_bytes),
                ("grpc.max_receive_message_length", self._config.max_message_bytes),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.min_time_between_pings_ms", self._config.http2_min_ping_interval_ms),
                # TODO: tune message limits based on production payloads.
            )
            channel = grpc.aio.insecure_channel(self._config.target, options=options)
            try:
                await asyncio.wait_for(channel.channel_ready(), timeout=self._config.connect_timeout)
            except asyncio.TimeoutError as exc:
                log_constant(self._logger, LOG_TRAINER_CLIENT_CONNECTION_TIMEOUT, message="Trainer daemon connection timeout", extra={"target": self._config.target})
                await channel.close()
                raise TrainerClientConnectionError(
                    f"Timed out waiting for trainer daemon at {self._config.target}"
                ) from exc
            except Exception:
                await channel.close()
                raise
            state.channel = channel
            state.stub = trainer_pb2_grpc.TrainerServiceStub(channel)
            log_constant(self._logger, LOG_TRAINER_CLIENT_CONNECTED, message="Trainer daemon connection established", extra={"target": self._config.target})
            return state.stub

    async def close(self) -> None:
        """Close all per-loop channels."""
        close_awaitables: list[asyncio.Future[Any]] = []
        for loop, state in list(self._loop_state.items()):
            channel = state.channel
            if channel is None:
                continue

            async def _close(ch: grpc.aio.Channel) -> None:
                await ch.close()

            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(_close(channel), loop)
                close_awaitables.append(asyncio.wrap_future(future))
            else:
                # Loop not running anymore; create a task in the current loop.
                task = asyncio.create_task(_close(channel))
                close_awaitables.append(task)

            state.channel = None
            state.stub = None

        if close_awaitables:
            await asyncio.gather(*close_awaitables, return_exceptions=True)

        self._loop_state.clear()

    async def submit_run(
        self,
        config_json: str,
        *,
        run_id: Optional[str] = None,
        deadline: Optional[float] = None,
    ) -> Any:
        stub = await self.ensure_connected()
        response = await stub.SubmitRun(
            trainer_pb2.SubmitRunRequest(run_id=run_id or "", config_json=config_json),
            timeout=deadline or self._config.deadline,
        )
        return response

    async def cancel_run(
        self,
        run_id: str,
        *,
        deadline: Optional[float] = None,
    ) -> Any:
        stub = await self.ensure_connected()
        return await stub.CancelRun(
            trainer_pb2.CancelRunRequest(run_id=run_id),
            timeout=deadline or self._config.deadline,
        )

    async def list_runs(
        self,
        statuses: Optional[Sequence[RunStatus]] = None,
        *,
        deadline: Optional[float] = None,
    ) -> Any:
        stub = await self.ensure_connected()
        status_filter = [_status_to_proto(status) for status in statuses] if statuses else []
        return await stub.ListRuns(
            trainer_pb2.ListRunsRequest(status_filter=status_filter),
            timeout=deadline or self._config.deadline,
        )

    @asynccontextmanager
    async def watch_runs(
        self,
        statuses: Optional[Sequence[RunStatus]] = None,
        *,
        deadline: Optional[float] = None,
        since_seq: int = 0,
    ) -> AsyncIterator[AsyncIterator[Any]]:
        stub = await self.ensure_connected()
        status_filter = [_status_to_proto(status) for status in statuses] if statuses else []
        call = stub.WatchRuns(
            trainer_pb2.WatchRunsRequest(status_filter=status_filter, since_seq=since_seq),
            timeout=deadline,
        )
        try:
            yield call
        finally:
            await call.aclose()

    async def heartbeat(
        self,
        run_id: str,
        *,
        deadline: Optional[float] = None,
    ) -> Any:
        stub = await self.ensure_connected()
        return await stub.Heartbeat(
            trainer_pb2.HeartbeatRequest(run_id=run_id),
            timeout=deadline or self._config.deadline,
        )

    async def get_health(self, *, deadline: Optional[float] = None) -> Any:
        stub = await self.ensure_connected()
        return await stub.GetHealth(
            trainer_pb2.HealthCheckRequest(),
            timeout=deadline or self._config.deadline,
        )

    @asynccontextmanager
    async def stream_run_steps(
        self,
        run_id: str,
        *,
        since_seq: int = 0,
        deadline: Optional[float] = None,
    ) -> AsyncIterator[AsyncIterator[Any]]:
        """Stream telemetry steps for a specific run."""
        stub = await self.ensure_connected()
        call = stub.StreamRunSteps(
            trainer_pb2.StreamStepsRequest(run_id=run_id, since_seq=since_seq),
            timeout=deadline,
        )
        try:
            yield call
        finally:
            try:
                call.cancel()
            except Exception:
                pass

    @asynccontextmanager
    async def stream_run_episodes(
        self,
        run_id: str,
        *,
        since_seq: int = 0,
        deadline: Optional[float] = None,
    ) -> AsyncIterator[AsyncIterator[Any]]:
        """Stream telemetry episodes for a specific run."""
        stub = await self.ensure_connected()
        call = stub.StreamRunEpisodes(
            trainer_pb2.StreamEpisodesRequest(run_id=run_id, since_seq=since_seq),
            timeout=deadline,
        )
        try:
            yield call
        finally:
            try:
                call.cancel()
            except Exception:
                pass


def _status_to_proto(status: RunStatus) -> int:
    mapping = {
    RunStatus.PENDING: trainer_pb2.RunStatus.RUN_STATUS_PENDING,
    RunStatus.DISPATCHING: trainer_pb2.RunStatus.RUN_STATUS_DISPATCHING,
    RunStatus.RUNNING: trainer_pb2.RunStatus.RUN_STATUS_RUNNING,
    RunStatus.COMPLETED: trainer_pb2.RunStatus.RUN_STATUS_COMPLETED,
    RunStatus.FAILED: trainer_pb2.RunStatus.RUN_STATUS_FAILED,
    RunStatus.CANCELLED: trainer_pb2.RunStatus.RUN_STATUS_CANCELLED,
    }
    return mapping[status]


__all__ = ["TrainerClient", "TrainerClientConfig"]
