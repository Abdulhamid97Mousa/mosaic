from __future__ import annotations

"""gRPC service implementation for the trainer daemon."""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import secrets
import logging
import sqlite3
from typing import Any, AsyncIterator, Callable, Deque, Iterable, Mapping, Optional, cast

from google.protobuf.timestamp_pb2 import Timestamp
import grpc

from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.constants import format_episode_id
from gym_gui.telemetry import TelemetrySQLiteStore
from gym_gui.telemetry.events import Topic, TelemetryEvent
from gym_gui.telemetry.run_bus import get_bus
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
        RunStatus.INIT: trainer_pb2.RunStatus.RUN_STATUS_INIT,
        RunStatus.HANDSHAKE: trainer_pb2.RunStatus.RUN_STATUS_HSHK,
        RunStatus.READY: trainer_pb2.RunStatus.RUN_STATUS_RDY,
        RunStatus.EXECUTING: trainer_pb2.RunStatus.RUN_STATUS_EXEC,
        RunStatus.PAUSED: trainer_pb2.RunStatus.RUN_STATUS_PAUSE,
        RunStatus.FAULTED: trainer_pb2.RunStatus.RUN_STATUS_FAULT,
        RunStatus.TERMINATED: trainer_pb2.RunStatus.RUN_STATUS_TERM,
    }
    proto_status = mapping.get(status)
    if proto_status is None:
        _LOGGER.warning("Unknown RunStatus encountered during proto mapping", extra={"status": status})
        return trainer_pb2.RunStatus.RUN_STATUS_UNSPECIFIED
    return proto_status


def _proto_to_statuses(statuses: Iterable[int]) -> list[RunStatus]:
    mapping = {
        trainer_pb2.RunStatus.RUN_STATUS_INIT: RunStatus.INIT,
        trainer_pb2.RunStatus.RUN_STATUS_HSHK: RunStatus.HANDSHAKE,
        trainer_pb2.RunStatus.RUN_STATUS_RDY: RunStatus.READY,
        trainer_pb2.RunStatus.RUN_STATUS_EXEC: RunStatus.EXECUTING,
        trainer_pb2.RunStatus.RUN_STATUS_PAUSE: RunStatus.PAUSED,
        trainer_pb2.RunStatus.RUN_STATUS_FAULT: RunStatus.FAULTED,
        trainer_pb2.RunStatus.RUN_STATUS_TERM: RunStatus.TERMINATED,
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


class RunTelemetryBroadcaster:
    """Publish/subscribe hub for telemetry streamed from workers."""

    _QUEUE_MAXSIZE = 2048
    _STEP_HISTORY_LIMIT = 4096
    _EPISODE_HISTORY_LIMIT = 1024

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._step_history: dict[str, Deque[tuple[int, Any]]] = defaultdict(
            lambda: deque(maxlen=self._STEP_HISTORY_LIMIT)
        )
        self._episode_history: dict[str, Deque[tuple[int, Any]]] = defaultdict(
            lambda: deque(maxlen=self._EPISODE_HISTORY_LIMIT)
        )
        self._step_listeners: dict[str, set[asyncio.Queue[Any]]] = defaultdict(set)
        self._episode_listeners: dict[str, set[asyncio.Queue[Any]]] = defaultdict(set)
        self._step_seq = 0
        self._episode_seq = 0

    async def publish_step(self, message: Any) -> tuple[int, int]:
        if not message.run_id:
            raise ValueError("run_id is required for telemetry steps")
        async with self._lock:
            self._step_seq += 1
            seq_id = self._step_seq
            payload = trainer_pb2.RunStep()
            payload.CopyFrom(message)
            payload.seq_id = seq_id
            history = self._step_history[message.run_id]
            history.append((seq_id, payload))
            listeners = list(self._step_listeners.get(message.run_id, set()))
        dropped = self._dispatch(listeners, payload)
        return seq_id, dropped

    async def publish_episode(self, message: Any) -> tuple[int, int]:
        if not message.run_id:
            raise ValueError("run_id is required for telemetry episodes")
        async with self._lock:
            self._episode_seq += 1
            seq_id = self._episode_seq
            payload = trainer_pb2.RunEpisode()
            payload.CopyFrom(message)
            payload.seq_id = seq_id
            history = self._episode_history[message.run_id]
            history.append((seq_id, payload))
            listeners = list(self._episode_listeners.get(message.run_id, set()))
        dropped = self._dispatch(listeners, payload)
        return seq_id, dropped

    async def subscribe_steps(
        self, run_id: str, since_seq: int
    ) -> tuple[asyncio.Queue[Any], list[Any]]:
        if not run_id:
            raise ValueError("run_id is required")
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
        async with self._lock:
            self._step_listeners[run_id].add(queue)
            replay: list[Any] = []
            # Always replay historical data, not just when since_seq > 0
            # This ensures GUI receives all steps even if it subscribes after training starts
            for seq_id, payload in self._step_history.get(run_id, ()):  # type: ignore[arg-type]
                if since_seq == 0 or seq_id > since_seq:
                    replay_payload = trainer_pb2.RunStep()
                    replay_payload.CopyFrom(payload)
                    replay.append(replay_payload)
        return queue, replay

    async def subscribe_episodes(
        self, run_id: str, since_seq: int
    ) -> tuple[asyncio.Queue[Any], list[Any]]:
        if not run_id:
            raise ValueError("run_id is required")
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
        async with self._lock:
            self._episode_listeners[run_id].add(queue)
            replay: list[Any] = []
            # Always replay historical data, not just when since_seq > 0
            # This ensures GUI receives all episodes even if it subscribes after training starts
            for seq_id, payload in self._episode_history.get(run_id, ()):  # type: ignore[arg-type]
                if since_seq == 0 or seq_id > since_seq:
                    replay_payload = trainer_pb2.RunEpisode()
                    replay_payload.CopyFrom(payload)
                    replay.append(replay_payload)
        return queue, replay

    async def unsubscribe_steps(self, run_id: str, queue: asyncio.Queue[Any]) -> None:
        async with self._lock:
            listeners = self._step_listeners.get(run_id)
            if listeners is not None:
                listeners.discard(queue)
                if not listeners:
                    self._step_listeners.pop(run_id, None)

    async def unsubscribe_episodes(
        self, run_id: str, queue: asyncio.Queue[Any]
    ) -> None:
        async with self._lock:
            listeners = self._episode_listeners.get(run_id)
            if listeners is not None:
                listeners.discard(queue)
                if not listeners:
                    self._episode_listeners.pop(run_id, None)

    @staticmethod
    def _dispatch(listeners: list[asyncio.Queue[Any]], payload: Any) -> int:
        if not listeners:
            return 0
        dropped = 0
        for queue in listeners:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                dropped += 1
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    pass
        if dropped:
            _LOGGER.warning(
                "Telemetry consumer lagging; dropped updates",
                extra={"dropped": dropped, "payload_type": type(payload).__name__},
            )
        return dropped


class TrainerService(trainer_pb2_grpc.TrainerServiceServicer):
    def __init__(
        self,
        registry: RunRegistry,
        gpu_allocator: GPUAllocator,
        broadcaster: Optional[RunEventBroadcaster] = None,
        telemetry_broadcaster: Optional[RunTelemetryBroadcaster] = None,
        health_provider: Optional[Callable[[], Any]] = None,
        telemetry_store: Optional[TelemetrySQLiteStore] = None,
    ) -> None:
        self._registry = registry
        self._gpu_allocator = gpu_allocator
        self._broadcaster = broadcaster or RunEventBroadcaster()
        self._telemetry_broadcaster = telemetry_broadcaster or RunTelemetryBroadcaster()
        self._health_provider = health_provider or (lambda: trainer_pb2.HealthCheckResponse(healthy=False))
        self._telemetry_store = telemetry_store
        self._worker_sessions: dict[str, str] = {}
        self._worker_capabilities: dict[str, dict[str, Any]] = {}
        self._executing_runs: set[str] = set()
        _LOGGER.debug("TrainerService initialized")

    def _save_run_config(self, run_id: str, config_json: str) -> None:
        """Save training config to disk for audit trail and reproducibility."""
        try:
            from gym_gui.config.paths import VAR_TRAINER_DIR, ensure_var_directories

            ensure_var_directories()
            config_dir = VAR_TRAINER_DIR / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)

            config_file = config_dir / f"config-{run_id}.json"
            config_file.write_text(config_json, encoding="utf-8")
            _LOGGER.info(
                "Saved training config to disk",
                extra={"run_id": run_id, "config_file": str(config_file)},
            )
        except Exception as e:
            _LOGGER.warning(
                "Failed to save training config to disk",
                extra={"run_id": run_id, "error": str(e)},
            )

    # ------------------------------------------------------------------
    async def SubmitRun(self, request: trainer_pb2.SubmitRunRequest, context: grpc.aio.ServicerContext) -> trainer_pb2.SubmitRunResponse:  # type: ignore[override]
        _LOGGER.info("SubmitRun received", extra={"request_run_id": request.run_id, "payload_bytes": len(request.config_json or '')})
        try:
            config = validate_train_run_config(json.loads(request.config_json))
        except Exception as exc:  # pragma: no cover - validation errors
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        if request.run_id and request.run_id != config.metadata.run_id:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "run_id mismatch with configuration digest",
            )

        # Save config to disk for audit trail
        self._save_run_config(config.metadata.run_id, config.to_json())

        # Register run; returns existing run_id if digest already exists
        resolved_run_id = self._registry.register_run(
            config.metadata.run_id, config.to_json(), config.metadata.digest
        )
        
        # If digest already exists, return the existing run
        if resolved_run_id != config.metadata.run_id:
            _LOGGER.info(
                "Duplicate run submission detected; returning existing run_id",
                extra={"requested_run_id": config.metadata.run_id, "existing_run_id": resolved_run_id},
            )
            return trainer_pb2.SubmitRunResponse(run_id=resolved_run_id, digest=config.metadata.digest)

        requested = config.payload["resources"]["gpus"]["requested"]
        mandatory = config.payload["resources"]["gpus"]["mandatory"]
        try:
            reservation = self._gpu_allocator.reserve(config.metadata.run_id, requested, mandatory)
            self._registry.update_gpu_slots(config.metadata.run_id, reservation.slots)
        except GPUReservationError as exc:
            self._registry.update_run_outcome(
                run_id=config.metadata.run_id,
                status=RunStatus.TERMINATED,
                outcome="failed",
                failure_reason=str(exc),
            )
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))

        await self._broadcast(config.metadata.run_id)
        _LOGGER.info("SubmitRun accepted", extra={"run_id": config.metadata.run_id})
        return trainer_pb2.SubmitRunResponse(run_id=config.metadata.run_id, digest=config.metadata.digest)

    async def CancelRun(self, request: trainer_pb2.CancelRunRequest, context: grpc.aio.ServicerContext) -> trainer_pb2.CancelRunResponse:  # type: ignore[override]
        record = self._registry.get_run(request.run_id)
        if not record:
            await context.abort(grpc.StatusCode.NOT_FOUND, "run not found")
        if record.status == RunStatus.TERMINATED:
            return trainer_pb2.CancelRunResponse()

        self._registry.update_run_outcome(
            run_id=request.run_id,
            status=RunStatus.TERMINATED,
            outcome="canceled",
        )
        self._gpu_allocator.release_many([request.run_id])
        self._registry.update_gpu_slots(request.run_id, [])
        await self._broadcast(request.run_id)
        return trainer_pb2.CancelRunResponse()

    async def ListRuns(self, request: trainer_pb2.ListRunsRequest, context: grpc.aio.ServicerContext) -> trainer_pb2.ListRunsResponse:  # type: ignore[override]
        try:
            statuses = _proto_to_statuses(request.status_filter)
            runs = self._registry.load_runs(statuses if statuses else None)
            return trainer_pb2.ListRunsResponse(runs=[_record_to_proto(run, seq_id=0) for run in runs])
        except sqlite3.OperationalError as e:
            _LOGGER.error(f"Database error in ListRuns: {e}")
            # Return empty response instead of crashing - caller can retry
            return trainer_pb2.ListRunsResponse(runs=[])

    async def WatchRuns(self, request: trainer_pb2.WatchRunsRequest, context: grpc.aio.ServicerContext) -> AsyncIterator[trainer_pb2.RunRecord]:  # type: ignore[override]
        queue, replay = await self._broadcaster.subscribe(request.since_seq)
        status_filter_proto = set(request.status_filter)
        try:
            if request.since_seq == 0:
                try:
                    for run in self._registry.load_runs(_proto_to_statuses(request.status_filter) or None):
                        snapshot = _record_to_proto(run, seq_id=0)
                        if not status_filter_proto or snapshot.status in status_filter_proto:
                            yield snapshot
                except sqlite3.OperationalError as e:
                    _LOGGER.error(f"Database error in WatchRuns initial load: {e}")
                    # Continue to watch for updates even if initial load fails
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

    async def RegisterWorker(self, request: trainer_pb2.RegisterWorkerRequest, context: grpc.aio.ServicerContext) -> trainer_pb2.RegisterWorkerResponse:  # type: ignore[override]
        if not request.run_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "run_id is required")

        record = self._registry.get_run(request.run_id)
        if not record:
            await context.abort(grpc.StatusCode.NOT_FOUND, "run not found")
        if record.status == RunStatus.TERMINATED:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "run already terminated")

        accepted_version = request.proto_version or "MOSAIC/1.0"
        session_token = secrets.token_hex(16)

        self._worker_sessions[request.run_id] = session_token
        self._worker_capabilities[request.run_id] = {
            "worker_id": request.worker_id,
            "worker_kind": request.worker_kind,
            "proto_version": request.proto_version,
            "schema_id": request.schema_id,
            "schema_version": request.schema_version,
            "supports_pause": request.supports_pause,
            "supports_checkpoint": request.supports_checkpoint,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }

        _LOGGER.info(
            "Worker registered",
            extra={
                "run_id": request.run_id,
                "worker_id": request.worker_id or "",
                "worker_kind": request.worker_kind or "",
                "proto_version": request.proto_version or "",
                "schema_id": request.schema_id or "",
            },
        )

        self._registry.record_heartbeat(request.run_id)

        if record.status not in {RunStatus.READY, RunStatus.EXECUTING, RunStatus.PAUSED}:
            self._registry.update_status(request.run_id, RunStatus.READY)
            await self._broadcast(request.run_id)

        return trainer_pb2.RegisterWorkerResponse(
            accepted_version=accepted_version,
            session_token=session_token,
        )

    async def ControlStream(self, request_iterator: AsyncIterator[trainer_pb2.ControlEvent], context: grpc.aio.ServicerContext) -> AsyncIterator[trainer_pb2.ControlEvent]:  # type: ignore[override]
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "ControlStream is not enabled for FSM rollout")
        yield trainer_pb2.ControlEvent()  # pragma: no cover - unreachable

    async def GetHealth(self, request: trainer_pb2.HealthCheckRequest, context: grpc.aio.ServicerContext) -> trainer_pb2.HealthCheckResponse:  # type: ignore[override]
        response = self._health_provider()
        return response

    async def StreamRunSteps(self, request: trainer_pb2.StreamStepsRequest, context: grpc.aio.ServicerContext) -> AsyncIterator[trainer_pb2.RunStep]:  # type: ignore[override]
        _LOGGER.debug("StreamRunSteps subscribe", extra={"run_id": request.run_id, "since_seq": int(request.since_seq)})
        if not request.run_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "run_id is required")

        # Load historical steps from telemetry database first (only on initial subscribe)
        seq_counter = 0
        if self._telemetry_store and int(request.since_seq) == 0:
            try:
                # Query all recent steps and filter by run_id (encoded in episode_id as {run_id}-ep{N})
                all_steps = self._telemetry_store.recent_steps(limit=4096)
                for step in all_steps:
                    # Extract run_id from episode_id (format: {run_id}-ep{N})
                    parts = step.episode_id.split('-ep')
                    if len(parts) == 2 and parts[0] == request.run_id:
                        seq_counter += 1
                        # Extract episode index from episode_id
                        try:
                            episode_index = int(parts[1]) if parts[1].isdigit() else 0
                        except (ValueError, IndexError):
                            episode_index = 0
                        
                        # Convert StepRecord to RunStep proto
                        action_json = ''
                        if step.action is not None:
                            try:
                                action_json = json.dumps(step.action)
                            except (TypeError, ValueError):
                                action_json = str(step.action)
                        
                        observation_json = ''
                        if step.observation is not None:
                            try:
                                observation_json = json.dumps(step.observation)
                            except (TypeError, ValueError):
                                observation_json = str(step.observation)
                        
                        render_hint_json = ''
                        if step.render_hint:
                            try:
                                render_hint_json = json.dumps(step.render_hint)
                            except (TypeError, ValueError):
                                render_hint_json = str(step.render_hint)
                        
                        payload = trainer_pb2.RunStep(
                            run_id=request.run_id,
                            episode_index=episode_index,
                            step_index=step.step_index,
                            action_json=action_json,
                            observation_json=observation_json,
                            reward=float(step.reward),
                            terminated=step.terminated,
                            truncated=step.truncated,
                            policy_label=str(step.info.get('policy_label', '') if step.info else ''),
                            backend=str(step.info.get('backend', '') if step.info else ''),
                            seq_id=seq_counter,
                            agent_id=step.agent_id or '',
                            render_hint_json=render_hint_json,
                            frame_ref=step.frame_ref or '',
                            payload_version=step.payload_version,
                        )
                        if step.timestamp:
                            payload.timestamp.FromDatetime(step.timestamp)
                        _LOGGER.debug("StreamRunSteps replay yield from DB", extra={"run_id": request.run_id, "ep": episode_index, "step": step.step_index, "seq": seq_counter})
                        yield payload
            except Exception as exc:
                _LOGGER.warning("StreamRunSteps: failed to load from telemetry store", extra={"run_id": request.run_id, "error": str(exc)}, exc_info=True)

        # NEW: Pass seq_counter to broadcaster to prevent double-delivery
        # If DB replay emitted steps (seq_counter > 0), broadcaster will skip its replay
        # If DB was empty (seq_counter == 0), broadcaster will replay from in-memory history
        queue, replay = await self._telemetry_broadcaster.subscribe_steps(request.run_id, seq_counter)
        try:
            for payload in replay:
                _LOGGER.debug("StreamRunSteps replay yield", extra={"run_id": request.run_id, "seq": getattr(payload, 'seq_id', None)})
                yield payload
            while True:
                payload = await queue.get()
                _LOGGER.debug("StreamRunSteps yield", extra={"run_id": request.run_id, "ep": getattr(payload, 'episode_index', None), "step": getattr(payload, 'step_index', None), "seq": getattr(payload, 'seq_id', None), "agent_id": getattr(payload, 'agent_id', None)})
                yield payload
        except asyncio.CancelledError:
            _LOGGER.debug("StreamRunSteps cancelled", extra={"run_id": request.run_id})
            raise
        finally:
            await self._telemetry_broadcaster.unsubscribe_steps(request.run_id, queue)

    async def StreamRunEpisodes(self, request: trainer_pb2.StreamEpisodesRequest, context: grpc.aio.ServicerContext) -> AsyncIterator[trainer_pb2.RunEpisode]:  # type: ignore[override]
        _LOGGER.debug("StreamRunEpisodes subscribe", extra={"run_id": request.run_id, "since_seq": int(request.since_seq)})
        if not request.run_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "run_id is required")

        # Load historical episodes from telemetry database first
        seq_counter = 0
        if self._telemetry_store:
            try:
                episodes = self._telemetry_store.episodes_for_run(request.run_id)
                for episode in episodes:
                    seq_counter += 1
                    # Convert EpisodeRollup to RunEpisode proto
                    # Parse episode_index from episode_id (format: "run_id-ep0002")
                    eid = episode.episode_id or ""
                    episode_index = 0
                    if "-ep" in eid:
                        try:
                            episode_index = int(eid.split("-ep", 1)[1])
                        except ValueError:
                            pass  # Keep episode_index = 0 if parsing fails
                    payload = trainer_pb2.RunEpisode(
                        run_id=request.run_id,
                        episode_index=episode_index,
                        total_reward=episode.total_reward,
                        steps=episode.steps,
                        terminated=episode.terminated,
                        truncated=episode.truncated,
                        metadata_json=json.dumps(episode.metadata) if episode.metadata else '{}',
                        seq_id=seq_counter,
                        agent_id=episode.agent_id or '',
                    )
                    if episode.timestamp:
                        payload.timestamp.FromDatetime(episode.timestamp)
                    _LOGGER.debug("StreamRunEpisodes replay yield from DB", extra={"run_id": request.run_id, "ep": episode.episode_id, "seq": seq_counter})
                    yield payload
            except Exception as exc:
                _LOGGER.warning("StreamRunEpisodes: failed to load from telemetry store", extra={"run_id": request.run_id, "error": str(exc)})

        # NEW: Pass seq_counter to broadcaster to prevent double-delivery
        # If DB replay emitted episodes (seq_counter > 0), broadcaster will skip its replay
        # If DB was empty (seq_counter == 0), broadcaster will replay from in-memory history
        queue, replay = await self._telemetry_broadcaster.subscribe_episodes(request.run_id, seq_counter)
        try:
            for payload in replay:
                _LOGGER.debug("StreamRunEpisodes replay yield", extra={"run_id": request.run_id, "seq": getattr(payload, 'seq_id', None)})
                yield payload
            while True:
                payload = await queue.get()
                _LOGGER.debug("StreamRunEpisodes yield", extra={"run_id": request.run_id, "ep": getattr(payload, 'episode_index', None), "seq": getattr(payload, 'seq_id', None), "agent_id": getattr(payload, 'agent_id', None)})
                yield payload
        except asyncio.CancelledError:
            _LOGGER.debug("StreamRunEpisodes cancelled", extra={"run_id": request.run_id})
            raise
        finally:
            await self._telemetry_broadcaster.unsubscribe_episodes(request.run_id, queue)

    async def PublishRunSteps(self, request_iterator: AsyncIterator[trainer_pb2.RunStep], context: grpc.aio.ServicerContext) -> trainer_pb2.PublishTelemetryResponse:  # type: ignore[override]
        _LOGGER.debug("PublishRunSteps open")
        accepted = 0
        dropped = 0
        run_id: Optional[str] = None
        session_validated = False
        try:
            async for message in request_iterator:
                if not message.run_id:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "run_id is required for telemetry ingestion")
                if run_id is None:
                    if not self._registry.get_run(message.run_id):
                        await context.abort(grpc.StatusCode.NOT_FOUND, "run not registered")
                    run_id = message.run_id
                elif message.run_id != run_id:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "telemetry stream cannot change run_id")

                if not session_validated:
                    if run_id not in self._worker_sessions:
                        await context.abort(
                            grpc.StatusCode.FAILED_PRECONDITION,
                            "worker must RegisterWorker before publishing telemetry",
                        )
                    session_validated = True
                    if run_id not in self._executing_runs:
                        self._registry.update_status(run_id, RunStatus.EXECUTING)
                        self._executing_runs.add(run_id)
                        await self._broadcast(run_id)
                try:
                    _, dropped_now = await self._telemetry_broadcaster.publish_step(message)
                except ValueError as exc:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                else:
                    accepted += 1
                    dropped += dropped_now
                    if accepted <= 5:
                        _LOGGER.debug("PublishRunSteps accepted", extra={"run_id": run_id, "ep": int(getattr(message, 'episode_index', 0)), "step": int(getattr(message, 'step_index', 0)), "agent_id": getattr(message, 'agent_id', '')})
                    if self._telemetry_store:
                        record = self._step_from_proto(message)
                        self._telemetry_store.record_step(record)

                    # Publish to RunBus for db_sink and other subscribers
                    if run_id:
                        try:
                            bus = get_bus()
                            agent_id = getattr(message, 'agent_id', 'default')
                            payload_dict = {
                                'run_id': run_id,
                                'agent_id': agent_id,
                                'episode_index': int(getattr(message, 'episode_index', 0)),
                                'step_index': int(getattr(message, 'step_index', 0)),
                                'reward': float(getattr(message, 'reward', 0.0)),
                                'terminated': bool(getattr(message, 'terminated', False)),
                                'truncated': bool(getattr(message, 'truncated', False)),
                                'timestamp': getattr(message, 'timestamp', ''),
                                'action': int(getattr(message, 'action', -1)) if hasattr(message, 'action') else None,
                                'worker_id': getattr(message, 'worker_id', ''),
                            }
                            evt = TelemetryEvent(
                                topic=Topic.STEP_APPENDED,
                                run_id=run_id,
                                agent_id=agent_id,
                                seq_id=int(getattr(message, 'seq_id', -1)),
                                ts_iso=getattr(message, 'timestamp', ''),
                                payload=payload_dict,
                            )
                            bus.publish(evt)
                            _LOGGER.debug(
                                "Published STEP_APPENDED to RunBus from daemon",
                                extra={"run_id": run_id, "agent_id": agent_id, "seq_id": getattr(message, 'seq_id', -1)},
                            )
                        except Exception as e:
                            _LOGGER.warning(
                                "Failed to publish STEP_APPENDED to RunBus",
                                extra={"run_id": run_id, "error": str(e)},
                            )
        except asyncio.CancelledError:
            _LOGGER.debug("PublishRunSteps cancelled", extra={"accepted": accepted})
            raise
        _LOGGER.info("PublishRunSteps close", extra={"run_id": run_id, "accepted": accepted, "dropped": dropped})
        if run_id:
            self._executing_runs.discard(run_id)
        return trainer_pb2.PublishTelemetryResponse(accepted=accepted, dropped=dropped)

    async def PublishRunEpisodes(self, request_iterator: AsyncIterator[trainer_pb2.RunEpisode], context: grpc.aio.ServicerContext) -> trainer_pb2.PublishTelemetryResponse:  # type: ignore[override]
        _LOGGER.debug("PublishRunEpisodes open")
        accepted = 0
        dropped = 0
        run_id: Optional[str] = None
        session_validated = False
        try:
            async for message in request_iterator:
                if not message.run_id:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "run_id is required for telemetry ingestion")
                if run_id is None:
                    if not self._registry.get_run(message.run_id):
                        await context.abort(grpc.StatusCode.NOT_FOUND, "run not registered")
                    run_id = message.run_id
                elif message.run_id != run_id:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "telemetry stream cannot change run_id")

                if not session_validated:
                    if run_id not in self._worker_sessions:
                        await context.abort(
                            grpc.StatusCode.FAILED_PRECONDITION,
                            "worker must RegisterWorker before publishing telemetry",
                        )
                    session_validated = True
                    if run_id not in self._executing_runs:
                        self._registry.update_status(run_id, RunStatus.EXECUTING)
                        self._executing_runs.add(run_id)
                        await self._broadcast(run_id)
                try:
                    _, dropped_now = await self._telemetry_broadcaster.publish_episode(message)
                except ValueError as exc:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                else:
                    accepted += 1
                    dropped += dropped_now
                    if accepted <= 3:
                        _LOGGER.debug("PublishRunEpisodes accepted", extra={"run_id": run_id, "ep": int(getattr(message, 'episode_index', 0)), "agent_id": getattr(message, 'agent_id', '')})
                    if self._telemetry_store:
                        rollup = self._episode_from_proto(message)
                        # Persist episode to SQLite (durable path)
                        self._telemetry_store.record_episode(rollup, wait=True)

                    # Publish to RunBus for db_sink and other subscribers
                    if run_id:
                        try:
                            bus = get_bus()
                            agent_id = getattr(message, 'agent_id', 'default')
                            payload_dict = {
                                'run_id': run_id,
                                'agent_id': agent_id,
                                'episode_index': int(getattr(message, 'episode_index', 0)),
                                'total_reward': float(getattr(message, 'total_reward', 0.0)),
                                'steps': int(getattr(message, 'steps', 0)),
                                'terminated': bool(getattr(message, 'terminated', False)),
                                'truncated': bool(getattr(message, 'truncated', False)),
                                'timestamp': getattr(message, 'timestamp', ''),
                                'metadata_json': getattr(message, 'metadata_json', ''),
                                'worker_id': getattr(message, 'worker_id', ''),
                            }
                            evt = TelemetryEvent(
                                topic=Topic.EPISODE_FINALIZED,
                                run_id=run_id,
                                agent_id=agent_id,
                                seq_id=int(getattr(message, 'seq_id', -1)),
                                ts_iso=getattr(message, 'timestamp', ''),
                                payload=payload_dict,
                            )
                            bus.publish(evt)
                            _LOGGER.debug(
                                "Published EPISODE_FINALIZED to RunBus from daemon",
                                extra={"run_id": run_id, "agent_id": agent_id, "seq_id": getattr(message, 'seq_id', -1)},
                            )
                        except Exception as e:
                            _LOGGER.warning(
                                "Failed to publish EPISODE_FINALIZED to RunBus",
                                extra={"run_id": run_id, "error": str(e)},
                            )
        except asyncio.CancelledError:
            _LOGGER.debug("PublishRunEpisodes cancelled", extra={"accepted": accepted})
            raise
        _LOGGER.info("PublishRunEpisodes close", extra={"run_id": run_id, "accepted": accepted, "dropped": dropped})
        if run_id:
            self._executing_runs.discard(run_id)
        return trainer_pb2.PublishTelemetryResponse(accepted=accepted, dropped=dropped)

    # ------------------------------------------------------------------
    async def _broadcast(self, run_id: str) -> None:
        record = self._registry.get_run(run_id)
        if not record:
            return
        if record.status == RunStatus.TERMINATED:
            self._worker_sessions.pop(run_id, None)
            self._worker_capabilities.pop(run_id, None)
            self._executing_runs.discard(run_id)
        await self._broadcaster.publish(_record_to_proto(record))

    # ------------------------------------------------------------------
    def _step_from_proto(self, message: Any) -> StepRecord:
        episode_index = int(message.episode_index)
        run_id = message.run_id or ""
        worker_id = message.worker_id or None
        if run_id:
            episode_id = format_episode_id(run_id, episode_index, worker_id)
        else:
            episode_id = format_episode_id("legacy", episode_index, worker_id)
        timestamp = self._timestamp_from_proto(message)
        action_value = self._decode_action(message.action_json)
        observation = self._decode_json_field(message.observation_json, default=None)
        render_hint = self._decode_json_field(message.render_hint_json, default=None)
        render_payload = self._decode_json_field(message.render_payload_json, default=None)

        info_payload: dict[str, Any] = {}
        if message.policy_label:
            info_payload["policy_label"] = message.policy_label
        if message.backend:
            info_payload["backend"] = message.backend
        if message.episode_seed:
            info_payload.setdefault("episode_seed", int(message.episode_seed))
        if worker_id:
            info_payload.setdefault("worker_id", worker_id)

        return StepRecord(
            episode_id=episode_id,
            step_index=int(message.step_index),
            action=action_value,
            observation=observation,
            reward=message.reward,
            terminated=message.terminated,
            truncated=message.truncated,
            info=info_payload,
            timestamp=timestamp,
            render_payload=render_payload,
            agent_id=message.agent_id or None,
            render_hint=render_hint if isinstance(render_hint, Mapping) else None,
            frame_ref=message.frame_ref or None,
            payload_version=int(message.payload_version) if message.payload_version else 0,
            run_id=run_id or None,
            worker_id=worker_id,
        )

    def _episode_from_proto(self, message: Any) -> EpisodeRollup:
        episode_index = int(message.episode_index)
        run_id = message.run_id or ""
        worker_id = message.worker_id or None
        if run_id:
            episode_id = format_episode_id(run_id, episode_index, worker_id)
        else:
            episode_id = format_episode_id("legacy", episode_index, worker_id)
        metadata = self._decode_json_field(message.metadata_json, default={})
        if not isinstance(metadata, Mapping):
            metadata = {}
        # Extract game_id from metadata if available
        game_id = metadata.get("game_id") if isinstance(metadata, dict) else None
        return EpisodeRollup(
            episode_id=episode_id,
            total_reward=message.total_reward,
            steps=int(message.steps),
            terminated=message.terminated,
            truncated=message.truncated,
            metadata=dict(metadata),
            timestamp=self._timestamp_from_proto(message),
            agent_id=message.agent_id or None,
            run_id=message.run_id or None,
            game_id=game_id,
            worker_id=worker_id,
        )

    @staticmethod
    def _decode_json_field(value: str, *, default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            _LOGGER.warning("Failed to decode telemetry JSON field", extra={"value_preview": value[:64]})
            return default

    @staticmethod
    def _timestamp_from_proto(message: Any) -> datetime:
        if hasattr(message, "timestamp") and hasattr(message, "HasField") and message.HasField("timestamp"):
            dt = message.timestamp.ToDatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        return datetime.now(timezone.utc)

    def _decode_action(self, value: str) -> int | None:
        parsed = self._decode_json_field(value, default=None)
        if isinstance(parsed, bool):
            return 1 if parsed else 0
        if isinstance(parsed, (int, float)):
            return int(parsed)
        if isinstance(parsed, str):
            try:
                return int(parsed)
            except ValueError:
                return None
        return None
