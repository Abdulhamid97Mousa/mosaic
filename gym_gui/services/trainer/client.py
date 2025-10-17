from __future__ import annotations

"""Async client bridge between the Qt GUI and the trainer daemon."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Sequence, cast

import grpc

from .registry import RunStatus
from gym_gui.services.trainer.proto import trainer_pb2 as trainer_pb2_module
from gym_gui.services.trainer.proto import trainer_pb2_grpc

trainer_pb2 = cast(Any, trainer_pb2_module)

_DEFAULT_DEADLINE = 10.0
_DEFAULT_CONNECT_TIMEOUT = 5.0
_DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


@dataclass(slots=True)
class TrainerClientConfig:
    target: str = "127.0.0.1:50055"
    deadline: float = _DEFAULT_DEADLINE
    keepalive_time: float = 30.0
    keepalive_timeout: float = 10.0
    connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT
    max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES


class TrainerClientConnectionError(RuntimeError):
    """Raised when the client cannot establish a gRPC channel."""


class TrainerClient:
    """High-level async API used by controllers to reach the daemon."""

    def __init__(self, config: Optional[TrainerClientConfig] = None) -> None:
        self._config = config or TrainerClientConfig()
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[trainer_pb2_grpc.TrainerServiceStub] = None
        self._lock = asyncio.Lock()

    async def ensure_connected(self) -> trainer_pb2_grpc.TrainerServiceStub:
        async with self._lock:
            if self._stub is not None and self._channel is not None:
                return self._stub
            options = (
                ("grpc.keepalive_time_ms", int(self._config.keepalive_time * 1000)),
                ("grpc.keepalive_timeout_ms", int(self._config.keepalive_timeout * 1000)),
                ("grpc.max_send_message_length", self._config.max_message_bytes),
                ("grpc.max_receive_message_length", self._config.max_message_bytes),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.min_time_between_pings_ms", 10_000),
                # TODO: tune message limits based on production payloads.
            )
            channel = grpc.aio.insecure_channel(self._config.target, options=options)
            try:
                await asyncio.wait_for(channel.channel_ready(), timeout=self._config.connect_timeout)
            except asyncio.TimeoutError as exc:
                await channel.close()
                raise TrainerClientConnectionError(
                    f"Timed out waiting for trainer daemon at {self._config.target}"
                ) from exc
            except Exception:
                await channel.close()
                raise
            self._channel = channel
            self._stub = trainer_pb2_grpc.TrainerServiceStub(channel)
            return self._stub

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None

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
            await call.aclose()

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
            await call.aclose()


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
