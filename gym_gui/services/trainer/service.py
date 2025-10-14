from __future__ import annotations

"""gRPC service implementation for the trainer daemon."""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import logging
from typing import Any, AsyncIterator, Callable, Deque, Iterable, Optional, cast

from google.protobuf.timestamp_pb2 import Timestamp
import grpc

from gym_gui.services.trainer import (
    GPUAllocator,
    GPUReservationError,
    RunRecord,
    RunRegistry,
    RunStatus,
    validate_train_run_config,
)
from gym_gui.services.trainer.proto import trainer_pb2 as trainer_pb2_module, trainer_pb2_grpc

trainer_pb2 = cast(Any, trainer_pb2_module)

_LOGGER = logging.getLogger("gym_gui.trainer.service")


def _timestamp(dt: Optional[datetime]) -> Optional[Timestamp]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ts = Timestamp()
    ts.FromDatetime(dt)
    return ts


def _record_to_proto(record: RunRecord, *, seq_id: Optional[int] = None):
    message = trainer_pb2.RunRecord(
        run_id=record.run_id,
        status=_status_to_proto(record.status),
        digest=record.digest,
    )
    if ts := _timestamp(record.created_at):
        message.created_at.CopyFrom(ts)
    if ts := _timestamp(record.updated_at):
        message.updated_at.CopyFrom(ts)
    if ts := _timestamp(record.last_heartbeat):
        message.last_heartbeat.CopyFrom(ts)
    if record.gpu_slot is not None:
        message.gpu_slot = record.gpu_slot
    if record.failure_reason:
        message.failure_reason = record.failure_reason
    if record.gpu_slots:
        message.gpu_slots.extend(record.gpu_slots)
    if seq_id is not None:
        message.seq_id = seq_id
    return message


def _status_to_proto(status: RunStatus):
    mapping = {
        RunStatus.PENDING: trainer_pb2.RunStatus.RUN_STATUS_PENDING,
        RunStatus.DISPATCHING: trainer_pb2.RunStatus.RUN_STATUS_DISPATCHING,
        RunStatus.RUNNING: trainer_pb2.RunStatus.RUN_STATUS_RUNNING,
        RunStatus.COMPLETED: trainer_pb2.RunStatus.RUN_STATUS_COMPLETED,
        RunStatus.FAILED: trainer_pb2.RunStatus.RUN_STATUS_FAILED,
        RunStatus.CANCELLED: trainer_pb2.RunStatus.RUN_STATUS_CANCELLED,
    }
    proto_status = mapping.get(status)
    if proto_status is None:
        _LOGGER.warning("Unknown RunStatus encountered during proto mapping", extra={"status": status})
        return trainer_pb2.RunStatus.RUN_STATUS_UNSPECIFIED
    return proto_status


def _proto_to_statuses(statuses: Iterable[int]) -> list[RunStatus]:
    mapping = {
        trainer_pb2.RunStatus.RUN_STATUS_PENDING: RunStatus.PENDING,
        trainer_pb2.RunStatus.RUN_STATUS_DISPATCHING: RunStatus.DISPATCHING,
        trainer_pb2.RunStatus.RUN_STATUS_RUNNING: RunStatus.RUNNING,
        trainer_pb2.RunStatus.RUN_STATUS_COMPLETED: RunStatus.COMPLETED,
        trainer_pb2.RunStatus.RUN_STATUS_FAILED: RunStatus.FAILED,
        trainer_pb2.RunStatus.RUN_STATUS_CANCELLED: RunStatus.CANCELLED,
    }
    result: list[RunStatus] = []
    for status in statuses:
        mapped = mapping.get(status)
        if mapped:
            result.append(mapped)
        elif status != trainer_pb2.RunStatus.RUN_STATUS_UNSPECIFIED:
            _LOGGER.warning("Unknown proto RunStatus encountered during mapping", extra={"status": status})
    return result


class RunEventBroadcaster:
    """Fan-out queue for broadcasting run updates to watching clients."""

    _QUEUE_MAXSIZE = 1024
    _HISTORY_LIMIT = 2048

    def __init__(self) -> None:
        self._listeners: set[asyncio.Queue[Any]] = set()
        self._lock = asyncio.Lock()
        self._history: Deque[tuple[int, Any]] = deque(maxlen=self._HISTORY_LIMIT)
        self._history_by_run: dict[str, Deque[tuple[int, Any]]] = defaultdict(lambda: deque(maxlen=self._HISTORY_LIMIT))
        self._seq_id = 0

    async def publish(self, record: Any) -> int:
        async with self._lock:
            self._seq_id += 1
            seq_id = self._seq_id
            message = trainer_pb2.RunRecord()
            message.CopyFrom(record)
            message.seq_id = seq_id
            self._history.append((seq_id, message))
            self._history_by_run[message.run_id].append((seq_id, message))
            pruned_listeners = 0
            for queue in self._listeners:
                if queue.full():
                    pruned_listeners += 1
                    # To prevent a slow consumer from blocking others, we drop the oldest
                    # message and enqueue the new one.
                    try:
                        queue.get_nowait()
                        queue.put_nowait(message)
                    except asyncio.QueueEmpty:
                        # The queue was full, but another consumer emptied it in the meantime.
                        # We can try to enqueue again.
                        try:
                            queue.put_nowait(message)
                        except asyncio.QueueFull:
                            # This is a rare race condition, we will just drop the message.
                            pass
                    except asyncio.QueueFull:
                        # This is a rare race condition, we will just drop the message.
                        pass
                else:
                    try:
                        queue.put_nowait(message)
                    except asyncio.QueueFull:
                        # This is a rare race condition, we will just drop the message.
                        pruned_listeners += 1

            if pruned_listeners > 0:
                _LOGGER.warning(
                    "Dropped messages for slow consumers",
                    extra={
                        "run_id": message.run_id,
                        "seq_id": seq_id,
                        "pruned_listener_count": pruned_listeners,
                        "total_listener_count": len(self._listeners),
                    },
                )
            return seq_id

    async def subscribe(self, since_seq: int = 0) -> tuple[asyncio.Queue[Any], list[Any]]:
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
        async with self._lock:
            self._listeners.add(queue)
            replay: list[Any] = []
            if since_seq > 0:
                for seq_id, message in self._history:
                    if seq_id > since_seq:
                        replay_message = trainer_pb2.RunRecord()
                        replay_message.CopyFrom(message)
                        replay.append(replay_message)
        return queue, replay

    async def unsubscribe(self, queue: asyncio.Queue[Any]) -> None:
        async with self._lock:
            self._listeners.discard(queue)


class TrainerService(trainer_pb2_grpc.TrainerServiceServicer):
    def __init__(
        self,
        registry: RunRegistry,
        gpu_allocator: GPUAllocator,
        broadcaster: Optional[RunEventBroadcaster] = None,
        health_provider: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._registry = registry
        self._gpu_allocator = gpu_allocator
        self._broadcaster = broadcaster or RunEventBroadcaster()
        self._health_provider = health_provider or (lambda: trainer_pb2.HealthCheckResponse(healthy=False))

    # ------------------------------------------------------------------
    async def SubmitRun(self, request: trainer_pb2.SubmitRunRequest, context: grpc.aio.ServicerContext) -> trainer_pb2.SubmitRunResponse:  # type: ignore[override]
        try:
            config = validate_train_run_config(json.loads(request.config_json))
        except Exception as exc:  # pragma: no cover - validation errors
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        if request.run_id and request.run_id != config.metadata.run_id:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "run_id mismatch with configuration digest",
            )

        # Register run; returns existing run_id if digest already exists
        actual_run_id = self._registry.register_run(
            config.metadata.run_id, config.to_json(), config.metadata.digest
        )
        
        # If digest already exists, return the existing run
        if actual_run_id != config.metadata.run_id:
            _LOGGER.info(
                "Duplicate run submission detected; returning existing run_id",
                extra={"requested_run_id": config.metadata.run_id, "existing_run_id": actual_run_id},
            )
            return trainer_pb2.SubmitRunResponse(run_id=actual_run_id, digest=config.metadata.digest)

        requested = config.payload["resources"]["gpus"]["requested"]
        mandatory = config.payload["resources"]["gpus"]["mandatory"]
        try:
            reservation = self._gpu_allocator.reserve(config.metadata.run_id, requested, mandatory)
            self._registry.update_gpu_slots(config.metadata.run_id, reservation.slots)
        except GPUReservationError as exc:
            self._registry.update_status(config.metadata.run_id, RunStatus.FAILED, failure_reason=str(exc))
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))

        await self._broadcast(config.metadata.run_id)
        return trainer_pb2.SubmitRunResponse(run_id=config.metadata.run_id, digest=config.metadata.digest)

    async def CancelRun(self, request: trainer_pb2.CancelRunRequest, context: grpc.aio.ServicerContext) -> trainer_pb2.CancelRunResponse:  # type: ignore[override]
        record = self._registry.get_run(request.run_id)
        if not record:
            await context.abort(grpc.StatusCode.NOT_FOUND, "run not found")
        if record.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return trainer_pb2.CancelRunResponse()

        self._registry.update_status(request.run_id, RunStatus.CANCELLED)
        self._gpu_allocator.release_many([request.run_id])
        self._registry.update_gpu_slots(request.run_id, [])
        await self._broadcast(request.run_id)
        return trainer_pb2.CancelRunResponse()

    async def ListRuns(self, request: trainer_pb2.ListRunsRequest, context: grpc.aio.ServicerContext) -> trainer_pb2.ListRunsResponse:  # type: ignore[override]
        statuses = _proto_to_statuses(request.status_filter)
        runs = self._registry.load_runs(statuses if statuses else None)
        return trainer_pb2.ListRunsResponse(runs=[_record_to_proto(run, seq_id=0) for run in runs])

    async def WatchRuns(self, request: trainer_pb2.WatchRunsRequest, context: grpc.aio.ServicerContext) -> AsyncIterator[trainer_pb2.RunRecord]:  # type: ignore[override]
        queue, replay = await self._broadcaster.subscribe(request.since_seq)
        status_filter_proto = set(request.status_filter)
        try:
            if request.since_seq == 0:
                for run in self._registry.load_runs(_proto_to_statuses(request.status_filter) or None):
                    snapshot = _record_to_proto(run, seq_id=0)
                    if not status_filter_proto or snapshot.status in status_filter_proto:
                        yield snapshot
            if replay:
                for message in replay:
                    if not status_filter_proto or message.status in status_filter_proto:
                        yield message
            while True:
                record = await queue.get()
                if not status_filter_proto or record.status in status_filter_proto:
                    yield record
        except asyncio.CancelledError:
            _LOGGER.debug("WatchRuns cancelled")
            raise
        finally:
            await self._broadcaster.unsubscribe(queue)

    async def Heartbeat(self, request: trainer_pb2.HeartbeatRequest, context: grpc.aio.ServicerContext) -> trainer_pb2.HeartbeatResponse:  # type: ignore[override]
        record = self._registry.get_run(request.run_id)
        if not record:
            await context.abort(grpc.StatusCode.NOT_FOUND, "run not found")
        self._registry.record_heartbeat(request.run_id)
        await self._broadcast(request.run_id)
        return trainer_pb2.HeartbeatResponse()

    async def GetHealth(self, request: trainer_pb2.HealthCheckRequest, context: grpc.aio.ServicerContext) -> trainer_pb2.HealthCheckResponse:  # type: ignore[override]
        response = self._health_provider()
        return response

    # ------------------------------------------------------------------
    async def _broadcast(self, run_id: str) -> None:
        record = self._registry.get_run(run_id)
        if not record:
            return
        await self._broadcaster.publish(_record_to_proto(record))
