# /home/hamid/Desktop/Projects/GUI_BDI_RL/gym_gui/services/trainer_daemon.py

from __future__ import annotations

"""Async trainer daemon responsible for orchestrating trainer runs."""

import argparse
import asyncio
import contextlib
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import signal
from typing import Any, Optional, TYPE_CHECKING

try:  # pragma: no cover - platform dependent
    import fcntl  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]
    import msvcrt  # type: ignore[import]

import grpc
from google.protobuf import __version__ as protobuf_version
from packaging.version import Version

from gym_gui.config.paths import (
    VAR_TELEMETRY_DIR,
    VAR_TRAINER_DIR,
    VAR_TRAINER_DB,
    ensure_var_directories,
)
from gym_gui.logging_config.log_constants import LOG_DAEMON_START
from gym_gui.logging_config.logger import configure_logging
from gym_gui.services.trainer import GPUAllocator, RunRegistry, RunStatus, TrainerDispatcher
from gym_gui.services.trainer.service import _record_to_proto
from google.protobuf.timestamp_pb2 import Timestamp

from gym_gui.services.trainer.proto import trainer_pb2, trainer_pb2_grpc
from gym_gui.services.trainer.registry import WALCheckpointStats
from gym_gui.telemetry import TelemetrySQLiteStore
from gym_gui.telemetry.db_sink import TelemetryDBSink
from gym_gui.telemetry.run_bus import get_bus

if TYPE_CHECKING:
    HealthCheckResponseType = Any  # pragma: no cover - typing alias
else:  # pragma: no cover - alias only used for runtime
    HealthCheckResponseType = getattr(trainer_pb2, "HealthCheckResponse")
from gym_gui.services.trainer.service import (
    RunEventBroadcaster,
    RunTelemetryBroadcaster,
    TrainerService,
)

_LOGGER = logging.getLogger("gym_gui.trainer.daemon")


class SingletonDaemonError(RuntimeError):
    """Raised if another trainer daemon instance already holds the lock."""


class _LockFile:
    """Advisory file lock used to ensure a single daemon instance.

    On POSIX systems this relies on :func:`fcntl.flock` and is advisory only. On
    Windows we fall back to ``msvcrt.locking`` which wraps the C runtime _locking
    function (equivalent to Win32 LockFileEx byte-range locking). Both provide
    non-blocking exclusive locks suitable for daemon singleton enforcement.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        ensure_var_directories()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o664)
        try:
            if fcntl:  # POSIX
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt:  # Windows
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        except (IOError, BlockingIOError) as exc:
            os.close(fd)
            raise SingletonDaemonError(
                f"Trainer daemon already running (lockfile: {self._path})"
            ) from exc
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if fcntl:  # POSIX
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            elif msvcrt:  # Windows
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        finally:
            os.close(self._fd)
            self._fd = None


class _PIDFile:
    """Best-effort PID file guard with stale process cleanup."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._owned_by_self = False

    def write(self) -> None:
        ensure_var_directories()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        current_pid = os.getpid()
        existing = self.read()
        if existing and existing != current_pid and self._is_process_alive(existing):
            raise SingletonDaemonError(
                f"Trainer daemon already running (pid file: {self._path}, pid: {existing})"
            )
        if existing and not self._is_process_alive(existing):
            _LOGGER.warning("Removing stale trainer PID file", extra={"pid": existing, "path": str(self._path)})
            with contextlib.suppress(FileNotFoundError):
                self._path.unlink()
        self._path.write_text(str(current_pid), encoding="utf-8")
        self._owned_by_self = True

    def read(self) -> Optional[int]:
        try:
            content = self._path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if not content:
            return None
        try:
            return int(content)
        except ValueError:
            return None

    def remove(self) -> None:
        if not self._owned_by_self:
            return
        with contextlib.suppress(FileNotFoundError):
            self._path.unlink()
        self._owned_by_self = False

    def _is_process_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # process exists but owned elsewhere
            return True
        except OSError:
            return False
        return True


class TrainerDaemon:
    """Coordinates trainer run lifecycle and maintenance tasks."""

    def __init__(
        self,
        registry: RunRegistry,
        *,
        gpu_allocator: Optional[GPUAllocator] = None,
        checkpoint_interval: int = 300,
        listen: str = "127.0.0.1:50055",
        max_message_bytes: int = 64 * 1024 * 1024,
        pid_file: Optional[_PIDFile] = None,
    ) -> None:
        self._registry = registry
        self._gpu_allocator = gpu_allocator or GPUAllocator(registry)
        self._checkpoint_interval = max(60, checkpoint_interval)
        self._stop_event = asyncio.Event()
        self._maintenance_task: Optional[asyncio.Task[None]] = None
        self._grpc_server: Optional[grpc.aio.Server] = None
        self._listen = listen
        self._broadcaster = RunEventBroadcaster()
        self._telemetry_broadcaster = RunTelemetryBroadcaster()
        self._dispatcher: Optional[TrainerDispatcher] = None
        self._pid_file = pid_file
        self._started_at: Optional[datetime] = None
        self._healthy = False
        self._fallback_full_streak = 0
        # gRPC server options - keepalive settings disabled to avoid immediate GOAWAY on idle connections
        self._grpc_options = (
            ("grpc.max_send_message_length", max_message_bytes),
            ("grpc.max_receive_message_length", max_message_bytes),
            # TODO: Re-enable keepalive once we understand why it causes GOAWAY on first connection
            # ("grpc.keepalive_time_ms", 30_000),
            # ("grpc.keepalive_timeout_ms", 10_000),
            # ("grpc.http2.max_pings_without_data", 0),
            # ("grpc.keepalive_permit_without_calls", 1),
            # ("grpc.http2.min_time_between_pings_ms", 10_000),
            # ("grpc.max_connection_idle_ms", 0),
        )
        ensure_var_directories()
        telemetry_db = VAR_TELEMETRY_DIR / "telemetry.sqlite"
        self._telemetry_store = TelemetrySQLiteStore(telemetry_db)

    async def run(self) -> None:
        _LOGGER.info("Trainer daemon starting")
        if self._pid_file:
            self._pid_file.write()
        self._started_at = datetime.now(timezone.utc)
        self._healthy = True
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop_event.set)
            except NotImplementedError:  # pragma: no cover - Windows fallback
                signal.signal(sig, lambda *_: self._stop_event.set())
        if self._listen.split(":", 1)[0] not in {"127.0.0.1", "localhost", "::1"}:
            _LOGGER.warning(
                "Trainer daemon gRPC endpoint is configured without TLS",
                extra={"listen": self._listen, "mode": "development-only"},
            )
        
        # Create dispatcher with broadcaster callback
        async def broadcast_callback(run_id: str) -> None:
            record = self._registry.get_run(run_id)
            if record:
                
                await self._broadcaster.publish(_record_to_proto(record))
        
        self._dispatcher = TrainerDispatcher(
            self._registry,
            self._gpu_allocator,
            broadcaster=broadcast_callback,
        )
        await self._dispatcher.start()

        # Initialize and start database sink for durable persistence
        # This ensures telemetry events are persisted to SQLite
        bus = get_bus()
        assert self._telemetry_store is not None, "Telemetry store must be initialized"
        db_sink = TelemetryDBSink(
            self._telemetry_store,
            bus,
            batch_size=64,
            checkpoint_interval=1000,
            writer_queue_size=512,
        )
        db_sink.start()
        _LOGGER.info("Telemetry database sink started in daemon")

        service = TrainerService(
            self._registry,
            self._gpu_allocator,
            broadcaster=self._broadcaster,
            telemetry_broadcaster=self._telemetry_broadcaster,
            health_provider=self._build_health_response,
            telemetry_store=self._telemetry_store,
        )
        self._grpc_server = grpc.aio.server(options=self._grpc_options)
        trainer_pb2_grpc.add_TrainerServiceServicer_to_server(service, self._grpc_server)
        # Development-only insecure transport. Provision TLS + auth when crossing hosts.
        self._grpc_server.add_insecure_port(self._listen)
        await self._grpc_server.start()
        _LOGGER.info("Trainer gRPC server listening", extra={"listen": self._listen})

        self._maintenance_task = asyncio.create_task(self._maintenance_loop(), name="trainer-maintenance")
        await self._stop_event.wait()
        _LOGGER.info("Stop signal received; shutting down")
        await self.shutdown()

    async def shutdown(self) -> None:
        self._healthy = False
        if self._dispatcher:
            await self._dispatcher.stop()
            self._dispatcher = None
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None
        if self._grpc_server:
            with contextlib.suppress(asyncio.CancelledError):
                await self._grpc_server.stop(5)
            self._grpc_server = None
        active_runs = self._registry.load_runs((RunStatus.RUNNING, RunStatus.DISPATCHING))
        self._gpu_allocator.release_many(run.run_id for run in active_runs)
        for run in active_runs:
            self._registry.update_gpu_slots(run.run_id, [])
        wal_stats = self._registry.wal_checkpoint()
        _LOGGER.info("Final WAL checkpoint", extra=self._wal_stats_extra(wal_stats, "TRUNCATE"))
        if self._pid_file:
            self._pid_file.remove()
        store = getattr(self, "_telemetry_store", None)
        if store is not None:
            try:
                store.close()
            finally:
                self._telemetry_store = None
        _LOGGER.info("Trainer daemon shutdown complete")

    async def _maintenance_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self._checkpoint_interval)
                except asyncio.TimeoutError:
                    # Always try to truncate the WAL file first.
                    truncate_stats = self._registry.wal_checkpoint("TRUNCATE")
                    extra = self._wal_stats_extra(truncate_stats, "TRUNCATE")

                    # If TRUNCATE was blocked by a reader (busy > 0) and left pages
                    # in the WAL, we can try a less aggressive checkpoint.
                    if truncate_stats.busy > 0 and truncate_stats.log_frames > 0:
                        self._fallback_full_streak += 1
                        level = logging.WARNING if self._fallback_full_streak >= 3 else logging.INFO
                        _LOGGER.log(
                            level,
                            "WAL TRUNCATE was blocked; will monitor.",
                            extra=extra,
                        )
                        # As a fallback, we could run a FULL checkpoint, but for now,
                        # we will just log and monitor the situation. A persistent
                        # high streak count indicates a stuck reader.
                    else:
                        self._fallback_full_streak = 0
                        _LOGGER.debug("WAL checkpoint", extra=extra)
                else:
                    break
        except asyncio.CancelledError:
            _LOGGER.debug("Maintenance loop cancelled")
            raise

    def _wal_stats_extra(self, stats: WALCheckpointStats, mode: str) -> dict[str, object]:
        remaining = max(stats.log_frames - stats.checkpointed_frames, 0)
        payload: dict[str, object] = {
            "checkpoint_mode": mode,
            "checkpoint_busy_readers": stats.busy,
            "checkpoint_pages_in_wal": stats.log_frames,
            "checkpoint_pages_checkpointed": stats.checkpointed_frames,
            "checkpoint_pages_remaining": remaining,
            "fallback_streak": self._fallback_full_streak,
        }
        return payload

    def _build_health_response(self) -> HealthCheckResponseType:
        message_ctor = getattr(trainer_pb2, "HealthCheckResponse")
        message = message_ctor(
            pid=os.getpid(),
            listen_address=self._listen,
            healthy=self._healthy and not self._stop_event.is_set(),
        )
        if self._started_at is not None:
            started_ts = Timestamp()
            started_ts.FromDatetime(self._started_at)
            message.started_at.CopyFrom(started_ts)
        return message


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gym GUI trainer daemon")
    parser.add_argument("--db", type=Path, default=VAR_TRAINER_DB, help="Path to the trainer registry SQLite database")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=300,
        help="Seconds between WAL checkpoints",
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=VAR_TRAINER_DIR / "trainer.lock",
        help="Path to lock file ensuring a single daemon instance",
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=VAR_TRAINER_DIR / "trainer.pid",
        help="PID file used for zombie detection and stale cleanup",
    )
    parser.add_argument(
        "--listen",
        default="127.0.0.1:50055",
        help="gRPC listen address (host:port). Use TLS when binding beyond loopback.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> None:
    configure_logging(level=getattr(logging, args.log_level))
    level = (
        LOG_DAEMON_START.level
        if isinstance(LOG_DAEMON_START.level, int)
        else getattr(logging, LOG_DAEMON_START.level)
    )
    _LOGGER.log(
        level,
        "%s %s",
        LOG_DAEMON_START.code,
        LOG_DAEMON_START.message,
        extra={"log_code": LOG_DAEMON_START.code, "stdout_redirected": True},
    )
    _LOGGER.info(
        "Trainer dependencies",
        extra={
            "grpc": grpc.__version__,
            "protobuf": protobuf_version,
            "required_grpc": "1.67.1",
            "required_protobuf": "5.27.2",
        },
    )
    if Version(grpc.__version__) < Version("1.67.1"):  # pragma: no cover - environment guard
        raise SystemExit("grpcio version 1.67.1 or higher is required")
    if Version(protobuf_version) < Version("5.27.2"):  # pragma: no cover - environment guard
        raise SystemExit("protobuf version 5.27.2 is required")
    lock = _LockFile(args.lock_file)
    pid_file = _PIDFile(args.pid_file)
    lock.acquire()
    try:
        registry = RunRegistry(str(args.db))
        daemon = TrainerDaemon(
            registry,
            checkpoint_interval=args.checkpoint_interval,
            listen=args.listen,
            pid_file=pid_file,
        )
        await daemon.run()
    finally:
        pid_file.remove()
        lock.release()


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    try:
        asyncio.run(_async_main(args))
    except SingletonDaemonError as exc:
        _LOGGER.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
